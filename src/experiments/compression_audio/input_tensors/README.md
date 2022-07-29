# Generating Experimental Data

This directory contains a subdirectory with a series of `*.npy` files that
contain tensors smaplef from the TinyML Keyword Spotting model. Available here:
https://github.com/mlcommons/tiny . These tensors are sampled from layers of
varying depth and used as the inputs to the software emulation in order to
produce results for the compression experiements. The input for the model used
in this experiment was constructed from the tensorflow speech commands dataset.
Creation of the input stream is handled by the `create_input_tensors_kws.py`
script.

While the raw data files are included in this repositor the scripts used to
obtain the data as well as the following instructions are included should
one wish to reproduce our results from scratch.

1. clone the TinyML repo: https://github.com/mlcommons/tiny
2. `cd $TINYML_ROOT/benchmark/training/keyword_spotting`
3. `cp $LEBUG_ROOT/src/experiments/compression_audio/input_tensors/create_input_tensors_kws.py .`
4. `python train.py` # note this step will download the tdfs speech commands dataset (multiple GB)
5. `python create_input_tensors_kws.py`
6. the input data files should be in the directory
   `$TINYML_ROOT/benchmark/training/keyword_spotting/input_tensors`
