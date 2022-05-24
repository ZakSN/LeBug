import sys
sys.path.insert(1, '../../src/')
from emulator.delta_compressor import DeltaCompressor
from emulator.trace_buffer import TraceBuffer
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import random
import numpy as np
import unittest

class TestCompression(unittest.TestCase):
    def build_frame(self, length, n, data_width, precision, p_compress=0.9, seed=1):
        '''
        Builds a frame (numpy array) of length, n element wide vectors. The
        elements of each vector are data_width bits long. p_compress
        sets the probability of two adjacent vectors containing elements that
        are at most precision bits different. i.e. p_compress sets the
        probability that two adjacent vectors are compressible
        '''
        np.random.seed(seed)
        random.seed(seed)

        data_max = twos_complement_max(data_width)
        data_min = twos_complement_min(data_width)
        delta_max = twos_complement_max(precision)
        # maximally negative delta reserved for INV symbol
        delta_min = twos_complement_min(precision) + 1

        frame = np.random.randint(data_min, data_max+1, (1, n))

        for v in range(length-1):
            if random.random() > p_compress:
                # add an uncompressible vector
                # in this case at least one element must be outside of +/- delta
                # of the previous vector

                delta_vector = np.random.randint(data_min-delta_min, data_max-delta_max + 1, (1, n))
                delta_vector = np.where(
                                   delta_vector>0,
                                   delta_vector+np.full((1,n), delta_max + 1),
                                   delta_vector-np.full((1,n), delta_min - 1)
                               )
            else:
                # add a compressible vector
                # in this case all elements must be within +/-delta of the
                # previous vector
                delta_vector = np.random.randint(delta_min, delta_max+1, (1,n))

            # truncate vector elements to data_wdith
            new_vector = np.clip(frame[-1] + delta_vector, data_min, data_max)
            frame = np.vstack([frame, new_vector])

        return frame

    def check_frame_equality(self, fin, fout):
        '''
        ensures that the vectors in the output are the same as the vectors in
        the input, and that they occur in the same order. Note that since the
        tracebuffer is circular there may be fewer output vectors than input
        vectors. This is not an error as long as it is only the old vectors that
        are lost.
        '''
        # the decompressed output can never be longer than the uncompressed
        # input
        if fout.shape[0] > fin.shape[0]:
            return False
        # vectors must have the same number of elements
        if fout.shape[1] != fin.shape[1]:
            return False

        # vectors must occur in the same order.
        # if the input is much longer than the tracebuffer (or if compression is
        # low) the tracebuffer may rollover many times, meaning that the sequence
        # of compressed data may not start with the first item in the input frame.
        # therefore we step through the input and look for a place where all of
        # the decompressed data matches.
        for i in range(fin.shape[0]):
            eq = np.all(fout[:] == fin[i:fout.shape[0]+i])
            if eq:
                return True
        return False

    def compression_ratio(self, v_nodata, compressed, decompressed):
        '''
        given a compressed tracebuffer and the decompressed data compute the
        compression ratio. since the compressed tracebuffer may not be full we
        need to know what the no data vector is so that tracebuffer addresses
        with no data are not counted
        '''
        # count one uncompressed value to account for last reg in the delta
        # compressor
        full_addrs = 1
        for i in range(compressed.shape[0]):
            if np.all(compressed[i] != v_nodata):
                full_addrs = full_addrs + 1

        # ideal compression ratio is DELTA_SLOTS:1
        return decompressed.shape[0]/full_addrs

    def basic_functionality(self,N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length):
        PRECISION = int(DATA_WIDTH/DELTA_SLOTS)
        INV = twos_complement_min(PRECISION)

        # instantiate hardware
        dc = DeltaCompressor(N,DATA_WIDTH,DELTA_SLOTS,INV)
        tb = TraceBuffer(N,TB_SIZE,DELTA_SLOTS,PRECISION,INV)

        # configure the decompression algorithm
        dd = DeltaDecompressor(N,DATA_WIDTH,DELTA_SLOTS,TB_SIZE)

        # create an input frame
        frame_in = self.build_frame(frame_length, N, DATA_WIDTH, PRECISION, p_compress=0.9)

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
        self.basic_functionality(1,4,8,4,256)

    def test_paramteric_sweep(self):
        # set parameters for sweep
        N_param = [1, 2, 4, 8, 16, 32]
        TB_SIZE_param = [4, 8, 16, 32, 64, 128, 256, 512]
        # note: trying to use 64bit numbers breaks numpy in annoying ways, since
        # some bitwise operations produce uint64 and others produce int64, whch
        # numpy will not do type coersion between. just ignoring for now
        DATA_WIDTH_param = [8, 16, 32,]# 64]
        DELTA_SLOTS_param = [2, 4, 8]
        frame_length_param = [4, 16, 64, 256, 1024]

        for N in N_param:
            for TB_SIZE in TB_SIZE_param:
                for DATA_WIDTH in DATA_WIDTH_param:
                    for DELTA_SLOTS in DELTA_SLOTS_param:
                        for frame_length in frame_length_param:
                            msg = "N="+str(N)+"\n"\
                                 +"TB_SIZE="+str(TB_SIZE)+"\n"\
                                 +"DATA_WIDTH="+str(DATA_WIDTH)+"\n"\
                                 +"DELTA_SLOTS="+str(DELTA_SLOTS)+"\n"\
                                 +"frame_length="+str(frame_length)+"\n"
                            with self.subTest(msg=msg):
                                self.basic_functionality(N,TB_SIZE,DATA_WIDTH,DELTA_SLOTS,frame_length)


if __name__=="__main__":
    unittest.main()
