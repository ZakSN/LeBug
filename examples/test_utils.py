import sys
sys.path.insert(1, '../../src/')
from emulator.delta_compressor import DeltaCompressor
from emulator.trace_buffer import TraceBuffer
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import random
import numpy as np

class TestUtils():
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

    def compression_ratio(self, v_nodata, compressed, decompressed, reg_count=1):
        '''
        given a compressed tracebuffer and the decompressed data compute the
        compression ratio. since the compressed tracebuffer may not be full we
        need to know what the no data vector is so that tracebuffer addresses
        with no data are not counted
        '''
        # count uncompresseds value to account for any registers in the delta
        # compression algorithm
        full_addrs = reg_count
        for i in range(compressed.shape[0]):
            if np.all(compressed[i] != v_nodata):
                full_addrs = full_addrs + 1

        # ideal compression ratio is DELTA_SLOTS:1
        return decompressed.shape[0]/full_addrs

