# Generating Experimental Data:

This directory contains two subdirectories, each with a series of `*.npy` files
that contain tensors that were sampled from the TinyML Visual Wake Words model
(mobileNet). Available here: https://github.com/mlcommons/tiny . These tensors
are sampled from layers of varying depth and used as the inputs to the software
emulation in order to produce results for the compression experiments. Each
subdirectory also includes the video file that was used as input for the model.

While the raw data files are included in this repository the scripts used to obtain
these data files as well as the following instructions are also included should
one whish to reproduce our results from scratch.

1. clone the TinyML repo: https://github.com/mlcommons/tiny
2. `cd $TINYML_ROOT/benchmark/training/visual_wake_words`
3. `cp $LEBUG_ROOT/src/experiments/compression_video/input_tensors/$EXAMPLE/video_96p.mp4 .`
3. `cp $LEBUG_ROOT/src/experiments/compression_video/input_tensors/create_input_tensors_vww.py .`
4. `python create_input_tensors.py`
5. the input data files should now be in the directory
   `$TINYML_ROOT/benchmark/training/visual_wake_words/input_tensors`
