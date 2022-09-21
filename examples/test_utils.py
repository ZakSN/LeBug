import sys
sys.path.insert(1, '../../src/')
from emulator.delta_compressor import DeltaCompressor
from emulator.trace_buffer import TraceBuffer
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import random
import numpy as np

class TestUtils():
    def build_frame(self, length, n, data_width, precision, p_compress=0.9, seed=1, limit=(None,None)):
        '''
        Builds a frame (numpy array) of length, n element wide vectors. The
        elements of each vector are data_width bits long. p_compress
        sets the probability of two adjacent vectors containing elements that
        are at most precision bits different. i.e. p_compress sets the
        probability that two adjacent vectors are compressible

        if the function is given a valid set of limits, ignore compressibility
        and just generate vectors uniformly in the range set by the limits
        '''
        np.random.seed(seed)
        random.seed(seed)

        delta_max = twos_complement_max(precision)
        # maximally negative delta reserved for INV symbol
        delta_min = twos_complement_min(precision) + 1
        if None in limit:
            data_max = twos_complement_max(data_width)
            data_min = twos_complement_min(data_width)
        else:
            p_compress = 2
            data_max = max(limit)
            data_min = min(limit)
            delta_max = max(limit)
            delta_min = min(limit)

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

            if None in limit:
                # truncate vector elements to data_wdith
                new_vector = np.clip(frame[-1] + delta_vector, data_min, data_max)
            else:
                # if we're given a limit ignore compressibility
                new_vector = delta_vector
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

    def compression_ratio(self, v_nodata, compressed, decompressed, bits_per_vec, correction=True):
        '''
        given a compressed tracebuffer and the decompressed data compute the
        compression ratio. since the compressed tracebuffer may not be full we
        need to know what the no data vector is so that tracebuffer addresses
        with no data are not counted
        '''
        full_addrs = 0
        for i in range(compressed.shape[0]):
            if np.all(compressed[i] != v_nodata):
                full_addrs = full_addrs + 1
        full_addrs += full_addrs/bits_per_vec # correct for compression flag bits

        # both the new and old models use compression flag bits, which we must
        # correct for, however, only the new model uses a last vector register
        if correction:
            full_addrs += 1 # correct for uncompressed last vector register

        # ideal compression ratio is DELTA_SLOTS:1
        return decompressed.shape[0]/full_addrs

    def worst_case_cr(self, v_nodata, compressed, bits_per_vec):
        C = 0
        for i in range(compressed.shape[0]):
            if np.all(compressed[i] != v_nodata):
                C += 1
        if C == 0:
            return 0
        wccr = 1/(1+(1/bits_per_vec)+(1/C))
        if wccr > 1.0:
            raise ValueError
        return wccr
