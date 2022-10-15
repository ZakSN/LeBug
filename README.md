# LeBug + Delta Compressor
LeBug is an open source FPGA debugging instrument initially developed by Daniel
Holanda Noronha et al. and published at FCCM in 2021. The original LeBug source
code is released under and MIT license and can be found
[here](https://github.com/danielholanda/LeBug).

This repository contains an extension to the original LeBug work. In particular,
it updates the LeBug pipeline to include an additional lossless compression
stage called the Delta Compressor. A paper discussing this compression stage and
its overall utility w.r.t. the initial instrument is to be published at
the [FPT 2022 conference](https://fpt22.hkust.edu.hk/).

## LeBug - Main Idea
A popular FPGA application is acceleration of Machine Learning tasks, such as
inference and training. However, both ML systems and FPGAs are complicated, and
as a result bugs are inevitable. Traditional signal based FPGA debugging
tactics are often not helpful when debugging ML systems or applications since
ML systems often operate on enourmous amounts of data, and have an internal
state that is not precisely specified by a designer (instead being the result
of training). As such, debug tools that endevour to reveal precise internal
state (e.g. by allowing a designer to view waveforms, or raw memory contents)
aren't much help.

In the world of software ML this limitation has been recognized, and tools such
as TensorFlow's Tensorboard have been created, which show a configurable
summarization of the application's internal state rather than a raw dump of the
contents of memory at a specific point in execution.

LeBug brings Tensorboard like debugging to FPGA ML applications. LeBug provides
a debugging instrument that can be attached to a main memory or bus in an
accelerator and used to compute statistics of the data captured on that interface.
These statistics are then stored for later review by a designer, in the hope
that a highlevel summarization of the ML application's internal state will
prove useful in understanding why it is behaving incorrectly.

An important point to note is that this process of summarizing internal application
state is effectively a form of extremely lossy compression. Rather than
attempting to repeatedly store the entire state of the device (i.e. exact values
of all tensors being processed at a specific point in time) a much smaller
summary is stored (i.e. a histogram showing a distribution of values in the
current tensor's being processed). This allows for more efficient use of on-chip
trace buffers, which store these summarizations.

## Delta Compressor - Main Idea
This repository integrates a new stage into the LeBug pipeline called the Delta
Compressor. The Delta Compressor losslessly compresses the summarizations that
the front-end of the pipeline produces before storing them in the trace buffer.
The utility of this extra compression step and implementation details are
addressed in the FPT paper associated with this project, but the core use case
is for debugging ML applications that are operating on a time varying input
with significant temporal redundancy, such as a video or audio stream.

# Repository Structure
The below provides a summary of the repository's directory structure.
```
\
    docs/ -- further documentaion (potentially somewhat out of date)
    examples/ -- test cases which demonstrate how to use the HW simulation, and  SW emulation
    	emu_test/ -- SW emulation test cases
	hw_test/ -- HW simulation test cases
    img/ -- Figrues used in documentation
    src/
    	containers/ -- Docker container to hold ModelSim environment for HW simulation
	emulator/ -- Python source code for the SW emulation
	experiments/ -- Source code for the experiments presented in the paper
		area_fmax/ -- code to generate quartus project to collect area and fmax data
		compression_audio/ -- audio compression experiment from paper
		compression_video/ -- video compression experiment from paper
		zlib_comparison/ -- experiment not included in FPT paper
	firmware/ -- Example firmware to configure the debug instrument
	hardware/ -- SystemVerilog source code for the HW implementation
	misc/ -- Test benches and repo-wide common code
	software/ -- Software data decompression program
```

# Using the Code
The following instructions where tested in a fresh Ubuntu 22.04 installation.
In order to follow along the following dependancies must be available on the
system (commands to install on Ubuntu):

- `python3` (`sudo apt install python3`)
- docker and the engine API (follow these instructions: https://docs.docker.com/engine/install/ubuntu/)
- python package `numpy` (`pip3 install numpy`)
- python package `docker` (`pip3 install docker`)
- python package `matplotlib` (`pip3 install matplotlib`)

## Running the test cases
To run the software emulator test cases do the following:
```
$ cd examples/emu_test
$ python3 emu_test.py
```
successful output is:
```
.......
----------------------------------------------------------------------
Ran 7 tests in 0.169s

OK
```
This test ensures basic functionality of the LeBug + DC pipeline

To run compression ratio tests do:
```
$ cd examples/emu_test
$ python3 compression_test.py
```
Note that this set of tests may take many minutes to complete. While running
the tests display many lines of floating point numbers (compression ratios).
successful output is:
```
.......
----------------------------------------------------------------------
Ran 7 tests in 0.169s

OK
```

The final script in the `emu_test` directory (`model_comparison.py`) compares
the Delta Compressor to an earlier software model. These tests are not required
to demonstrate functionality of this code base. If they are run expect some
tests to fail, and see comments in the script for further details.

Both the emulated 'basic functionality' and 'compression test' unit tests have
hardware counterparts, which can be run as follows:
```
$ cd examples/hw_test
$ python3 hw_test.py
```
Note that on the first run a docker image will be downloaded -- this may be
very slow depending on network speeds. Each test case will print some vectors
to the stdout while it is running. Successful output is:
```
.
----------------------------------------------------------------------
Ran 10 tests in 81.575s

OK
```
To run the hardware compression tests do the following:
```
$ cd examples/hw_test
$ python3 compression_test.py
```
This unit test also prints a lot of output, and may take many minutes to run,
since it needs to use ModelSim in a docker container. Successful output is:
```
.
----------------------------------------------------------------------
Ran 2 tests in 4173.958s

OK (skipped=1)
```

## Reproducing Audio Processing Model Results
This experiment runs a software simulation of the whole LeBug + DC flow over
a set of tensors that where captured from an CNN that processes an audio stream.
The specifics of how the internal tensors where extracted are described in
[this](src/experiments/compression_audio/input_tensors/README.md) file, which is
stored with the input data itself (i.e. the captured tensors).

To run the audio processing experiment:
```
$ cd src/experiments/compression_audio
$ python3 audio_compression_experiment.py
```
Note that this experiment may take many minutes to run. The results of the
experiment will be in a file called `audio_compression_results.csv`. In order
to plot these results run:
```
$ python3 plot_compression_results.py
```
which will produce a plot called `audio_compression_plot.png`. Note that this
plot does not appear in the final paper, but is discussed in Section IV-B.

## Reproducing Video Processing Model Results
The steps for running the video processing experiment are nearly the same as
in the audio case:
```
$ cd src/experiments/compression_video
$ python3 video_compression_experiment.py
$ python3 plot_compression_results.py
```
the plot `video_compression_plot.png` is Figure 6 from the report and shows the
data from the results file `video_compression_results.csv`. These results
are discussed in Section IV-B of the final paper.

Note that both the audio and video processing experiments are highly
parallelized, if possible committing many cores to run these experiments will
make them go *much* faster.

## Reproducing Area and Fmax results
The Area and Fmax results presented in the paper (Section IV-A) were produced
with Quartus 20.3 targeting a Stratix 10 1SG280LN2F43E1VG device. Installing
and configuring quartus is beyond the scope of this README, and as such the
following instructions will assume that the reader has a Quartus installation
available. Note that while it may be difficult to exactly reproduce the Area
and Fmax results provided in the paper (if for example a slightly different
version of quartus is used) the general conclusions should remain unchanged.

To generate a Quartus project, from which the Area and Fmax experiments can be
run do the following:
```
$cd src/experiments/area_fmax
$python3 build_quartus_project.py
```
This will generate a timestamped subdirectory in the current directory with a
name like: `yyymmddhhmm_xxxxxxx_lebug/` where `xxxxxxx` is the short hash of
the current HEAD. This subdirectory is self contained and may be copied to a
CAD server that does not have the LeBug + DC git repo on it. Within this
subdirectory is a script called `synthesis_test.py` running this script will
automatically run a synthesis of all of the configurations presented in the
paper, and collate the results in a report file.

Note that in order for the `synthesis_test.py` script to work, it must be
configured via the `build_quartus_project.py` script. In particular the location
of the Quartus binary on the synthesis system must be added to the
`build_quartus_project.py` script on line 21. The default string in the file
is just an example -- *it must be changed*.

Note In order to avoid run-to-run variability due to placer seeds changing the
`build_quartus_project.py` sets placer seeds to consistent values.

Finally, in order to generate the relative performance figures (figs 4 and 5 in
the paper) copy the report file produced by the synthesis script (named something
like `yyyymmddhhmm_xxxxxxx_summary.rpt`) to the directory
`src/experiments/area_fmax/results`, edit line 46 of
`src/experiments/area_fmax/results/datavis.py` to point at the new report, and
then run:
```
python3 datavis.py
```
the files `fmax_percent_diff.png` and `area_percent_diff.png` are the report
graphics.

# Authors

- Zakary Nafziger (zsnafzig-at-ece.ubc.ca)
- Martin Chua
- Daniel Holanda Noronha
- Steve Wilton

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
