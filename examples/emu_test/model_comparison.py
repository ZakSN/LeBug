from compression_utils import CompressionUtils
import sys
sys.path.insert(1, '../../src/')
from emulator.delta_compressor import DeltaCompressor
from emulator.trace_buffer import TraceBuffer
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import misc.initial_compression_model as model
import random
import numpy as np
import unittest

class TestCompressionModels(unittest.TestCase,CompressionUtils):
    def new_compression_ratio(self,frame_in,N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,INV,PRECISION,frame_length):
        '''
        Calculate the compression ratio for the new compression algorithm.
        note, when calculating compression ratio we unset reg_count. This is
        because
        '''
        # instantiate hardware
        dc = DeltaCompressor(N,DATA_WIDTH,DELTA_SLOTS,INV)
        tb = TraceBuffer(N,TB_SIZE,DELTA_SLOTS,PRECISION,INV)

        # configure the new decompression algorithm
        dd = DeltaDecompressor(N,DATA_WIDTH,DELTA_SLOTS,TB_SIZE)

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

        # get the new compression ratio
        v_nodata = np.full((1, N), n_bit_nodata(DELTA_SLOTS, PRECISION, INV))
        # XXX unsetting reg_count, bad?
        # new compression ratio slightly worse than old if TB very shallow,
        # and reg_count set, since counting the last_reg as uncompressed counts
        # for relatively more in the compression ratio
        return self.compression_ratio(v_nodata, tbuffer, frame_out, 1)

    def old_compression_ratio(self,frame_in,N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,INV,PRECISION,frame_length):
        '''
        Calculate the comrpession ratio for the old compression algorithm.
        The old algorithm has some cornercases
        1. unused addresses in the tracebuffer are set to 0 and therefore cannon
           be differentiated from addresses that contain only 0 deltas, without
           also analyzing the cfbuffer. Rather than adjust the compression ratio
           calculation function, we ensure that the input frame is always longer
           than what can theoretically be placed in the tracebuffer, thereby
           removing the need to differentiate between unpopulated tracebuffer
           addresses and all 0 deltas
        2. The old algorithm doesn't seem to handle the case where the data is
           perfectly compressible - in this case there is never an overflow
           and so no full precision values are stored and the data cannot be
           decompressed. As a result when compress_p=0 the compression ratio
           is calculated to be 0
        '''
        cfg = {
            'BUFFER_SIZE':TB_SIZE,
            'PRECISION':DELTA_SLOTS, # PRECISION is named differently
            'WIDTH':DATA_WIDTH,
            'DEBUG':0
        }

        frame_in = list(frame_in.flatten().astype(int))

        # feed the input frame into the model
        cfbuffer, tbuffer, tbptr, vis = model.packer(cfg, frame_in)

        # decompress the tracebuffer
        frame_out = model.unpacker(cfg, cfbuffer, tbuffer, tbptr)

        # reformat the lists as numpy arrays
        frame_out = np.array([np.array([i]) for i in frame_out])
        frame_in = np.array([np.array([i]) for i in frame_in])
        tbuffer = np.array([np.array([i]) for i in tbuffer])

        # get the new compression ratio
        v_nodata = np.full((1, N), n_bit_nodata(DELTA_SLOTS, PRECISION, INV))
        return self.compression_ratio(v_nodata, tbuffer, frame_out, 0)

    def comparison(self,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length,p):
        PRECISION = int(DATA_WIDTH/DELTA_SLOTS)
        INV = twos_complement_min(PRECISION)

        # create an input frame
        frame_in = self.build_frame(
            frame_length,
            1,
            DATA_WIDTH,
            PRECISION,
            p_compress=p
        )

        args = (
            frame_in,
            1,
            TB_SIZE,
            DATA_WIDTH,
            DELTA_SLOTS,
            INV,
            PRECISION,
            frame_length
        )

        new_cr = self.new_compression_ratio(*args)
        old_cr = self.old_compression_ratio(*args)
        print(np.array([new_cr, old_cr]))
        self.assertTrue(new_cr >= old_cr)

    @unittest.skip("skipping sweep")
    def test_param_config(self):
        '''
        dummy test for debugging specific parameter configurations
        '''
        self.comparison(32,8,4,4*32*8,0.0)

    def test_paramteric_sweep(self):
        '''
        This test is based on the function from compression_test
        Note that 16 out of 1080 subtests fail (~1.5%), i.e. the older algorithm
        performs slightly better. All of these cases occur when either the
        tracebuffer is very small or the input frame is nearly incompressible.
        This is due to the fact that the new algorithm stores a single
        uncompressed value in a register, that must be read out to decompress
        the tracebuffer. If this value is not counted in the compression ratio
        then all tests pass. Of the 16 failing tests only 5 fail by more than 5%
        (~0.5%)
        '''
        # set parameters for sweep
        # note N hardcoded to 1 to account for non-vectorized form of old model
        TB_SIZE_param = [4, 8, 16, 32, 64, 128, 256, 512]
        DATA_WIDTH_param = [8, 16, 32,]
        DELTA_SLOTS_param = [2, 4, 8]
        frame_length_param = [2, 4, 8]
        compression_prob = [0.0, 0.2, 0.5, 0.9, 1]

        for TB_SIZE in TB_SIZE_param:
            for DATA_WIDTH in DATA_WIDTH_param:
                for DELTA_SLOTS in DELTA_SLOTS_param:
                    for frame_length in frame_length_param:
                        for p in compression_prob:
                            msg = "TB_SIZE="+str(TB_SIZE)+"\n"\
                                 +"DATA_WIDTH="+str(DATA_WIDTH)+"\n"\
                                 +"DELTA_SLOTS="+str(DELTA_SLOTS)+"\n"\
                                 +"frame_length="+str(frame_length)+"\n"\
                                 +"p="+str(p)+"\n"
                            with self.subTest(msg=msg):
                                # setting framesize to always be larger than can
                                # theoretically be compressed in the tracebuffer
                                # to avoid a corner case in the old model
                                # see old_compression_ratio docstring
                                self.comparison(
                                    TB_SIZE,
                                    DATA_WIDTH,
                                    DELTA_SLOTS,
                                    DELTA_SLOTS*TB_SIZE*frame_length,
                                    p
                                )

if __name__ == "__main__":
    unittest.main()
