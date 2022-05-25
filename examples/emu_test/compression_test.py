from compression_utils import CompressionUtils
import sys
sys.path.insert(1, '../../src/')
from emulator.delta_compressor import DeltaCompressor
from emulator.trace_buffer import TraceBuffer
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import random
import numpy as np
import unittest

class TestCompression(unittest.TestCase, CompressionUtils):
    def basic_functionality(self,N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length,p):
        PRECISION = int(DATA_WIDTH/DELTA_SLOTS)
        INV = twos_complement_min(PRECISION)

        # instantiate hardware
        dc = DeltaCompressor(N,DATA_WIDTH,DELTA_SLOTS,INV)
        tb = TraceBuffer(N,TB_SIZE,DELTA_SLOTS,PRECISION,INV)

        # configure the decompression algorithm
        dd = DeltaDecompressor(N,DATA_WIDTH,DELTA_SLOTS,TB_SIZE)

        # create an input frame
        frame_in = self.build_frame(frame_length, N, DATA_WIDTH, PRECISION, p_compress=p)

        # feed the frame into the dc->tb emulator:
        for i in range(frame_in.shape[0]):
            compressed_data = dc.step((frame_in[i], 1))
            tb.step(compressed_data)

        # get state after the last cycle
        last_reg = dc.last_reg
        tbuffer  = tb.tbuffer
        cfbuffer = tb.cfbuffer
        tbptr    = tb.tbptr

        # decompress the tracebuffer
        frame_out = dd.decompress(last_reg, tbuffer, cfbuffer, tbptr)

        # ensure that the decompressed data matches the input data:
        self.assertTrue(self.check_frame_equality(frame_in, frame_out))

        v_nodata = np.full((1, N), n_bit_nodata(DELTA_SLOTS, PRECISION, INV))
        cr = self.compression_ratio(v_nodata, tbuffer, frame_out)
        print(cr)

        # ensure that the compressed data did not take more space than the
        # decompressed data
        self.assertTrue(cr >= 1)
        # ensure that the reported compression ratio does not exceed the theoretical
        # maximum
        self.assertTrue(cr <= DELTA_SLOTS)

    @unittest.skip("skipping sweep")
    def test_param_config(self):
        '''
        dummy test for debugging specific parameter configurations
        '''
        self.basic_functionality(1,4,8,4,256,0.9)

    def test_paramteric_sweep(self):
        '''
        This test is huge, and fairly slow. Conceptually it ought to easy to
        parallelize, but it seemsl ike the unit test framework wants to share
        state between subtests, which makes it annoying in practice.
        '''
        # set parameters for sweep
        N_param = [1, 2, 4, 8, 16, 32]
        TB_SIZE_param = [4, 8, 16, 32, 64, 128, 256, 512]
        # note: trying to use 64bit numbers breaks numpy in annoying ways, since
        # some bitwise operations produce uint64 and others produce int64, whch
        # numpy will not do type coersion between. just ignoring for now
        DATA_WIDTH_param = [8, 16, 32,]# 64]
        DELTA_SLOTS_param = [2, 4, 8]
        frame_length_param = [4, 16, 64, 256, 1024]
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
                                    self.basic_functionality(N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length,p)


if __name__=="__main__":
    unittest.main()
