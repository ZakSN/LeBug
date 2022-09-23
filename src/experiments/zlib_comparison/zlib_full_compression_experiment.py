import numpy as np
import os
import sys
sys.path.insert(1, '../')
from compression_experiment import multi_run
experiment_root = os.path.join('..','compression_video')
sys.path.insert(1, experiment_root)
os.chdir(experiment_root)
import video_compression_experiment as vce

# zlib compression does not vary with number of delta slots
delta_slots = vce.delta_slots[0:1]
RESULTS_FILE = os.path.join('..','zlib_comparison','zlib_video_compression_results.csv')

bpl_args = (vce.INPUT_TENSOR_DIR,
            RESULTS_FILE,
            vce.input_videos,
            vce.sampling_period,
            delta_slots,
            vce.firmwares,
            vce.N)

if __name__ == "__main__":
    zlib_off_proc, q = vce.build_proc_list(*bpl_args)
    zlib_on_proc = []
    for p in zlib_off_proc:
        zlib_on_proc.append((p[0], (*p[1], [9, None])))
    multi_run(zlib_on_proc, RESULTS_FILE, q)
