import sys
sys.path.insert(1, '../')
from test_utils import TestUtils
sys.path.insert(1, '../../src/')
from emulator.emulator import emulatedHw
from hardware.hardware import rtlHw
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import firmware.firmware as firm
import math
import numpy as np
np.set_printoptions(precision=3, suppress=False)
import unittest
import warnings

class TestCompressionHW(unittest.TestCase, TestUtils):
    def setUp(self):
        # squelch docker warnings
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)

    def generic_test(self,N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length,p):
         PRECISION = int(DATA_WIDTH/DELTA_SLOTS)
         INV = twos_complement_min(PRECISION)

         # configuration dictionary with fixed params
         cfg = {
             'N' : N,
             'TB_SIZE' : TB_SIZE,
             'DATA_WIDTH' : DATA_WIDTH,
             'DELTA_SLOTS' : DELTA_SLOTS,
             'M' : N,
             'IB_DEPTH' : frame_length,
             'FUVRF_SIZE' : 4,
             'VVVRF_SIZE' : 8,
             'MAX_CHAINS' : 4,
             'DATA_TYPE' : 'int',
             'BUILDING_BLOCKS' : ['InputBuffer', 'FilterReduceUnit', 'VectorVectorALU', 'VectorScalarReduce', 'DataPacker', 'DeltaCompressor', 'TraceBuffer'],
             'DEVICE_FAM' : 'Stratix 10',
         }

         # Instantiate hardware
         hw_proc = rtlHw(**cfg)

         # Configure the decompression algorithm
         dd = DeltaDecompressor(N,DATA_WIDTH,DELTA_SLOTS,TB_SIZE)

         # Create a frame
         frame_in = self.build_frame(frame_length, N, DATA_WIDTH, PRECISION, p_compress=p)

         # Push the frame into the input buffer
         for i in range(len(frame_in)):
             hw_proc.push([frame_in[i],False,False])

         # Configure firmware (RAW)
         fw = firm.raw(hw_proc.compiler)

         # Run the HW sim
         steps = int(1.1*frame_length) + 30
         hw_results = hw_proc.run(steps=steps,gui=False,log=False)

         # decompress the hardware results
         # XXX we also convert to signed numbers
         hw_results_compressed = [
             np.array(toInt([hw_results['dc']['last_vector_out'][-1]], DATA_WIDTH))[0], # last vector register
             np.array(toInt(hw_results['tb']['mem_data'], DATA_WIDTH)), # trace buffer
             np.array(toInt(hw_results['tb']['cfb'], DATA_WIDTH)), # compression flag buffer
             np.array(toInt(hw_results['tb']['tb_ptr_out']))[-1][0], # trace buffer pointer
         ]
         hw_results_decompressed = dd.decompress(*hw_results_compressed)

         # ensure that the decompressed data matches the input data
         print(frame_in)
         print(hw_results_decompressed)
         self.assertTrue(self.check_frame_equality(frame_in, hw_results_decompressed))

         # ensure that compressed data took less space than the decompressed data
         # we've converted all of the numbers from the tb to signed values, including
         # the nodata symbol (which is properly a bit vector and not a number...)
         # so we need to do some extra math here
         unsigned_nodata = n_bit_nodata(DELTA_SLOTS, PRECISION, INV)
         signed_nodata = unsigned_nodata - 2**(DATA_WIDTH)
         v_nodata = np.full((1, N), signed_nodata)
         cr = self.compression_ratio(v_nodata, hw_results_compressed[1], hw_results_decompressed)
         self.assertTrue(cr >= 1)

         # ensure that the compression ratio does not exceed the theoretical max
         self.assertTrue(cr <= DELTA_SLOTS)

    @unittest.skip("")
    def test_parameters(self):
        N = 8
        TB_SIZE = 32
        DATA_WIDTH = 8
        DELTA_SLOTS = 8
        frame_length = 16
        p = 0.2
        self.generic_test(N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length,p)

    def test_parametric_sweep(self):
        '''
        sweep various hardware parameters and ensure that compression still
        works. Slow (~1 hr) since each combination requires starting a docker
        container, and running a simulation.
        '''
        # set sweep parameters
        # XXX: upstream pipeline is not designed to handle vectors of width 1
        N_param = [2, 8, 32]
        TB_SIZE_param = [4, 32, 256]
        DATA_WIDTH_param = [8, 16, 32]
        DELTA_SLOTS_param = [2, 4, 8]
        frame_length_param = [16, 64, 512]
        compression_prob = [0.0, 0.2, 0.5, 0.9, 1]

        for N in N_param:
            for TB_SIZE in TB_SIZE_param:
                for DATA_WIDTH in DATA_WIDTH_param:
                    for DELTA_SLOTS in DELTA_SLOTS_param:
                        for frame_length in frame_length_param:
                            for p in compression_prob:
                                msg = "N="+str(N)+"\n"\
                                     +"TB_SIZE="+str(TB_SIZE)+"\n"\
                                     +"DATA_WIDTH="+str(DATA_WIDTH)+"\n"\
                                     +"DELTA_SLOTS="+str(DELTA_SLOTS)+"\n"\
                                     +"frame_length="+str(frame_length)+"\n"\
                                     +"p="+str(p)+"\n"
                                with self.subTest(msg=msg):
                                    self.generic_test(N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length,p)

if __name__ == "__main__":
    unittest.main()
