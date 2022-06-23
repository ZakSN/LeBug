import os
#from vww_model import mobilenet_v1
import tensorflow as tf
assert tf.__version__.startswith('2')
from keras import Model
import numpy as np
import cv2
import sys

INPUT_VIDEO = "./video_96p.mp4"
FRAMES = os.path.splitext(INPUT_VIDEO)[0] + "_frames"
MODEL_PATH = "trained_models/vww_96.h5"
TENSORS = "input_tensors"

def get_images(path_to_video, output_folder):
    cam = cv2.VideoCapture(path_to_video)
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    except OSError:
        print('Error: Creating directory of data')
    # frame
    currentframe = 0
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = os.path.join(output_folder, str(currentframe) + '.jpg')
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

# split the input video into frames
get_images(INPUT_VIDEO, FRAMES)

# get the visual wake words model
vww_model = tf.keras.models.load_model(MODEL_PATH)

# pick some places to tap the model
tap_points = np.linspace(0, len(vww_model.layers)-1, 5, dtype=int)

input_data = tf.keras.utils.image_dataset_from_directory(
    FRAMES,
    labels = None,
    batch_size = 1,
    shuffle = False,
    image_size = (96, 96)
)

# for each tap point save the output tensor
try:
    if not os.path.exists(TENSORS):
        os.makedirs(TENSORS)
except OSError:
    print('Error: could not create tensor data directory')

for tap in tap_points:
    tapped_model = Model(inputs=vww_model.inputs, outputs=vww_model.layers[tap].output)
    out = []
    for image in input_data:
        out.append(tapped_model.predict(image))

    np.save(os.path.join(TENSORS, vww_model.layers[tap].name), np.array(out))
