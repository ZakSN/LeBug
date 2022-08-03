import tensorflow as tf
import numpy as np
import os
import wave
import kws_util
import get_dataset
import shutil
import keras_model

def ds_from_tfr(datadir):
    filenames = os.listdir(datadir)
    filenames = [os.path.join(datadir,f) for f in filenames if 'tfrecord' in f]
    return tf.data.TFRecordDataset(filenames)
    

def write_wav(file_name, audio, frame_rate=16000, channels=1, sample_width=2, 
              data_width=8, endianness='little'):
    '''
    write the data in the np array 'audio' to a wav file with name 'file_name'
    - sample_width is the number of bytes per wav file sample
    - data_width is the number of bytes per item in the the audio iterable
    defaults write a 16bit wav file from an array of int64s
    '''
    def condition(i, dw, sw):
        if endianness == 'little':
            return (i % dw) < sw
        if endianness == 'big':
            return (i % dw) >= (dw-sw)
    audio = audio.tobytes()
    downscaled = []
    for i in range(len(audio)):
        if condition(i, data_width, sample_width):
            downscaled.append(audio[i])
    audio = bytes(downscaled)
    with wave.open(file_name, 'wb') as wavfile:
        wavfile.setparams((channels, sample_width, frame_rate, 0, 'NONE', 'NONE'))
        wavfile.writeframes(audio)

def build_stream(nclips, datadir, seed=1):
    '''
    randomly appends nclips from the speech commands dataset and returns a
    numpy array contaning all of the samples
    '''
    raw_dataset = ds_from_tfr(datadir)

    raw_dataset = raw_dataset.shuffle(nclips*seed, seed=seed)

    stream = None
    labels = []
    for e in raw_dataset.take(nclips):
        example = tf.train.Example()
        example.ParseFromString(e.numpy())
        result = {}
        for key, feature in example.features.feature.items():
          kind = feature.WhichOneof('kind')
          result[key] = np.array(getattr(feature, kind).value)
        if stream is None:
            stream = result['audio']
        else:
            stream = np.append(stream, result['audio'])
        labels.append(result['label'])

    return stream, labels

def resample_stream(stream, stride, window, workdir=None):
    '''
    cuts the stream in to blocks of samples of length 'window' spaced with a 
    stride of 'stride' samples. if workdir is defined writes wav files to
    workdir for debug, otherwise returns the list of samples.
    '''
    resampled = []
    start_idx = 0
    end_idx = start_idx + window
    while end_idx < stream.size:
        resampled.append(stream[start_idx:end_idx])
        start_idx = start_idx + stride
        end_idx = start_idx + window

    if workdir is None:
        return resampled

    # destroy the workdir if it exists
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        pass

    # create a new workdir
    os.makedirs(workdir)

    for idx, clip in enumerate(resampled):
        write_wav(os.path.join(workdir, 'sample_' + str(idx) + '.wav'), clip)

def create_dataset(samples, workdir):
    '''
    This is a poorly designed function
    the preprocessing step for the kws network expects a tensorflow speech
    commands dataset. after preprocessing this dataset can be fed into the
    model to train or evaluate it. We want to run the model on new data, however.
    To avoid rebuilding the preprocessing pipeline we need to format our
    new data exactly like the tensorflow speech commands dataset. Ideally
    this would all be done in memory, but I couldn't figure out a clean way to
    do this, so unfortunately we need to hit the disk...
    '''
    def create_example(label, audio):
        def _int64_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))
        feature = {
            'label' : _int64_feature(label),
            'audio' : _int64_feature(audio.tolist()),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def parse_tfrecord(t):
        return tf.io.parse_single_example(t, {
            'label' : tf.io.FixedLenFeature([], tf.int64),
            'audio' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        })

    # if the workdir doesn't exist create it:
    try:
        if not os.path.exists(workdir):
            os.makedirs(workdir)
    except OSError:
        pass

    # write the samples to TFRecords in the workdir
    with tf.io.TFRecordWriter(os.path.join(workdir, 'kws_resampled.tfrecord')) as writer:
        for s in samples:
            # we don't care about inference results so label everything as _unknown_
            writer.write(create_example([11], s).SerializeToString())

    # read TFRecords into a dataset
    ds = ds_from_tfr(workdir)

    # parse the raw dataset
    ds = ds.map(parse_tfrecord)

    return ds

if __name__ == '__main__':
    workdir = 'kws_experiment_work'
    tensors = 'input_tensors'

    clip_length = 16000
    for stride in np.linspace(0, clip_length, 5, dtype=int)[1:]:
        Flags, unparsed = kws_util.parse_command()
        # merge multiple audio clips into a single stream, resample it with the
        # provided stride and window, and return the resulting data packaged as a
        # tensorflow dataset
        stream, labels = build_stream(10, os.path.join(Flags.data_dir, 'speech_commands', '0.0.2'), seed=1)
        resampled = resample_stream(stream, stride, clip_length)
        ds = create_dataset(resampled, workdir)

        # configure and run the audio preprocessing steps on the dataset created above
        label_count = 11
        model_settings = keras_model.prepare_model_settings(label_count, Flags)
        bg_path=Flags.bg_path
        BACKGROUND_NOISE_DIR_NAME='_background_noise_'
        background_data = get_dataset.prepare_background_data(bg_path,BACKGROUND_NOISE_DIR_NAME)
        ds = ds.map(get_dataset.get_preprocess_audio_func(model_settings, is_training=False, background_data=background_data))
        ds = ds.map(get_dataset.convert_dataset)
        ds = ds.batch(1)

        # tap the model and save internal tensors
        kws = tf.keras.models.load_model('trained_models/kws_model.h5')

        try:
            if not os.path.exists(tensors):
                os.makedirs(tensors)
        except OSError:
            print('Error: could not make tensor data directory')

        tap_points = np.linspace(0, len(kws.layers)-1, 5, dtype=int)
        for tap in tap_points:
            tapped_model =tf.keras.Model(inputs=kws.inputs, outputs=kws.layers[tap].output)
            out = []
            for e in ds:
                x = tapped_model.predict(e[0])
                out.append(x)
            np.save(os.path.join(tensors, kws.layers[tap].name+"_"+str(stride)+"_"+str(clip_length)), np.array(out))
