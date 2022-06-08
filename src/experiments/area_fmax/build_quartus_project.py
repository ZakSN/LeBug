import numpy as np
import datetime
import os
import shutil
import sys
import subprocess
from distutils.dir_util import copy_tree
sys.path.insert(1, '../../')
from hardware.hardware import rtlHw
sys.path.remove('../../')
'''
This script generates a quartus project and associated python script required
to reproduce our experimental results. The generated directory structure
is such that it can be copied to a different computer (e.g. a CAD server) or
run in place, provided python3.6 or higher is available, and the quartus binary
is available as set in QUARTUS_BIN
'''

# change to point at $QUARTUS_INSTALL/quartus
# quartus 20.3 pro was used for published results
QUARTUS_BIN="/home/danielhn/intelFPGA_pro/20.3/quartus"

# part number must match hw_cfg[DEVICE_FAM]
# defualt is a stratix 10 part
PART_NUMBER = "1SG280LN2F43E1VG"

hw_cfg = {
    # these values are swept in the experimental project, but we need initial values
    # to generate the hardware
    'N' : 8,
    'M' : 4,
    # default hardware configuration
    'IB_DEPTH' : 32,
    'FUVRF_SIZE' : 4,
    'VVVRF_SIZE' : 8,
    'TB_SIZE' : 8,
    'MAX_CHAINS' : 4,
    'DATA_TYPE' : 'fixed_point',
    'DATA_WIDTH' : 32,
    'BUILDING_BLOCKS' : ['InputBuffer','FilterReduceUnit','VectorVectorALU','VectorScalarReduce','DataPacker','TraceBuffer'],
    'DEVICE_FAM' : 'Stratix 10',
}

# fitter seeds used to produce FCCM '21 results
# access as seeds[N][M]
seeds = {}
seeds[16]  = {1:800, 4:793, 16:341}
seeds[32]  = {1:864, 2:659, 8:437, 32:374}
seeds[64]  = {1:654, 4:197, 16:684, 64:755}
seeds[128] = {1:554, 8:386, 32:900, 128:857}

# to use random seeds
# seeds = lambda n, m: None

# project is generated with the name <timestamp>_<current commit>_lebug/
PROJ_DIR = datetime.datetime.now().strftime("%Y%m%d%H%M") + "_" + subprocess.check_output("git rev-parse --short HEAD", shell=True).decode("utf-8").replace("\n", "") + "_lebug"

BATCH_RUN = "synthesis_test.py"

os.mkdir(PROJ_DIR)

# produce the experimental configuration
np.save(
    os.path.join(PROJ_DIR, "config.npy"),
    np.array([
        QUARTUS_BIN,
        list([[16, 1, 0], [16, 1, 0], [16, 4, 0], [16, 16, 0], [32, 1, 0], [32, 2, 0], [32, 8, 0], [32, 32, 0], [64, 1, 0], [64, 4, 0], [64, 16, 0], [64, 64, 0], [128, 1, 0], [128, 8, 0], [128, 32, 0], [128, 128, 0], [128, 32, 0]]),
        'mlDebug',
        seeds
    ], dtype=object),
    allow_pickle=True)

# copy the batch run script
shutil.copyfile(os.path.join("resources", BATCH_RUN), os.path.join(PROJ_DIR, BATCH_RUN))

# create the quartus project directory
os.mkdir(os.path.join(PROJ_DIR, "quartus_project"))

# create the RTL
hw = rtlHw(**hw_cfg)
hw.generateRtl()

# copy the RTL over into the project
copy_tree('rtl', os.path.join(PROJ_DIR, "quartus_project"))

# get rid of the external rtl directory
shutil.rmtree('rtl')

# copy memory ip over
shutil.copyfile(os.path.join("resources", "dual_port_ram.ip"), os.path.join(PROJ_DIR, "quartus_project", "dual_port_ram.ip"))

# copy top file over
shutil.copyfile(os.path.join("resources", "top.sv"), os.path.join(PROJ_DIR, "quartus_project", "top.sv"))

# copy sdc file over
shutil.copyfile(os.path.join("resources", "mlDebug.sdc"), os.path.join(PROJ_DIR, "quartus_project", "mlDebug.sdc"))

# create quartus project files from the templates
with open(os.path.join("resources", "template.mlDebug.qpf"), 'r') as file:
    data = file.readlines()

data[25] = data[25].replace('XXX', datetime.datetime.now().strftime("%H:%m:%S %B %-d, %Y"))

with open(os.path.join(PROJ_DIR, "quartus_project", "mlDebug.qpf"), 'w') as file:
    file.writelines(data)

with open(os.path.join("resources", "template.mlDebug.qsf"), 'r') as file:
    data = file.readlines()

data[2] = data[2].replace('XXX', datetime.datetime.now().strftime("%H:%m:%S %B %-d, %Y").upper())
data[7] = data[7].replace('XXX', PART_NUMBER)
data[8] = data[8].replace('XXX', hw_cfg['DEVICE_FAM'])

with open(os.path.join(PROJ_DIR, "quartus_project", "mlDebug.qsf"), 'w') as file:
    file.writelines(data)
