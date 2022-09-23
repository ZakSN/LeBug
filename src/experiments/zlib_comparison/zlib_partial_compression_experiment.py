import numpy as np
import os
import sys
sys.path.insert(1, '../')
from compression_experiment import multi_run
experiment_root = os.path.join('..','compression_video')
sys.path.insert(1, experiment_root)
os.chdir(experiment_root)
import video_compression_experiment as vce

# we subsample the experiment space to reduce the number of cases we test, since
# each test requires compressing many partial trace buffers

RESULTS_FILE = os.path.join('..','zlib_comparison','zlib_video_partial_results.csv')
delta_slots = vce.delta_slots[0:1]

input_videos = vce.input_videos[0:1]
sampling_period = vce.sampling_period[0:1]
#firmwares = [i for i in vce.firmwares if 'raw' not in i.__name__]

partial_cr_results = os.path.join('..','zlib_comparison','partial_cr_results')

bpl_args = (vce.INPUT_TENSOR_DIR,
            RESULTS_FILE,
            input_videos,
            sampling_period,
            delta_slots,
            vce.firmwares,
            vce.N)

if __name__ == '__main__':
    proc_list, q = vce.build_proc_list(*bpl_args)
    new_proc_list = []
    for p in proc_list:
        rf = os.path.join(partial_cr_results, '_'.join(map(str, p[1][3]))+'.csv')
        # only add this experiment if the corresponding results file does not
        # exist (i.e. this config has not been run)
        if not os.path.exists(rf):
            new_proc_list.append((p[0], (*p[1], [9, rf], 20000)))
    multi_run(new_proc_list, RESULTS_FILE, q)
