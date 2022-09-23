import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

class DeltaCompressor():
    def __init__(self,N,DATA_WIDTH,DELTA_SLOTS,INV):
        self.N = N
        self.DATA_WIDTH = DATA_WIDTH
        self.DELTA_SLOTS = DELTA_SLOTS

        # define some additional symbols
        self.PRECISION = int(self.DATA_WIDTH/self.DELTA_SLOTS)
        self.INV = INV
        self.DELTA_MAX = twos_complement_max(self.PRECISION)
        # smallest possible delta is reserved as an 'invalid' symbol
        self.DELTA_MIN = twos_complement_min(self.PRECISION) + 1
        self.DATA_MAX = twos_complement_max(self.DATA_WIDTH)
        self.DATA_MIN = twos_complement_min(self.DATA_WIDTH)

        # initial delta compressor state
        self.ptr = 0 # compression register pointer
        self.comp_reg = np.full((1, self.N), 0) # compression register
        self.last_reg = np.full((1, self.N), 0) # uncompressed last vector register
        self.first_cycle = True # first cycle is a special case

        # entropy measurment
        self.measure_entropy = False
        self.bi = 0 # number of input bits that are 1
        self.ti = 0 # total number of input bits
        self.bo = 0 # number of output bits that are 1
        self.to = 0 # total number of output bits
        self.pop_mask = ((2**self.DATA_WIDTH)-1)

        self.num_vec = 0

    def pop_count(self, v):
        # pop_count algo from: https://stackoverflow.com/a/843846
        # adjusted to mask to bitwidth, to handle -ve 2's c numbers
        def pc(n):
            c = 0
            while n & self.pop_mask:
                c +=1
                n &= n - 1
            return c
        vpc = np.vectorize(pc)
        return vpc(v).sum()

    def step(self, packed_data):
        v_in, v_in_valid = packed_data

        # vector ouput signals
        v_out_valid = False
        v_out_comp  = None
        v_out       = None
        inc_tb_ptr  = False

        # if the input is not valid bail out early
        if v_in_valid == 0:
            return v_out_valid, v_out_comp, v_out, inc_tb_ptr

        self.num_vec += 1

        if self.measure_entropy:
            self.ti = self.ti + self.N*self.DATA_WIDTH
            self.bi = self.bi + self.pop_count(v_in)

        assert_vector_size(v_in, self.N, self.DATA_MIN, self.DATA_MAX)

        # first cycle should only fill the last vector register
        if self.first_cycle:
            self.first_cycle = False
            self.last_reg = v_in
            return v_out_valid, v_out_comp, v_out, inc_tb_ptr

        # compute the delta
        delta = self.last_reg - v_in

        # check for overflow
        overflow = False
        try:
            assert_vector_size(delta, self.N, self.DELTA_MIN, self.DELTA_MAX)
        except AssertionError:
            overflow = True

        if not overflow:
            # incoming vector can be compressed into the comp_reg

            # place older deltas into more significant bits
            mask = np.full(self.N, -1, dtype=int)
            mask = mask << int((self.DELTA_SLOTS - self.ptr)*self.PRECISION)
            self.comp_reg = np.bitwise_and(mask, self.comp_reg)

            mask = np.full(self.N, -1, dtype=int)
            mask = mask << int(self.PRECISION)
            mask = np.bitwise_not(mask)
            delta = np.bitwise_and(mask, delta)

            # shift the delta into place
            self.comp_reg = np.bitwise_or(self.comp_reg,
                (delta << int((self.DELTA_SLOTS - self.ptr - 1)*self.PRECISION)))

            # set all of the empty slots in the compression register to INV(alid)
            for i in range(0, (self.DELTA_SLOTS - self.ptr - 1)):
                inv_vector = np.bitwise_and(mask, np.full(self.N, self.INV, dtype=int))
                inv_vector = inv_vector << int(i*self.PRECISION)
                self.comp_reg = np.bitwise_or(self.comp_reg, inv_vector)

            # set output signals:
            v_out_valid = True
            v_out_comp  = True
            v_out       = self.comp_reg
            if self.ptr == 0:
                inc_tb_ptr = True
            else:
                inc_tb_ptr = False

            # update the compression register pointer
            self.ptr = rollover_counter(self.ptr, self.DELTA_SLOTS)
        else:
            # incoming vector cannot be compressed (overflow)
            # therefore write the last vector as full precison
            v_out_valid = True
            v_out_comp  = False
            v_out       = self.last_reg
            inc_tb_ptr  = True

            # reset the compression register pointer
            self.ptr = 0

        if v_out_valid and self.measure_entropy:
            self.to = self.to + self.N*self.DATA_WIDTH
            self.bo = self.bo + self.pop_count(v_out)

        self.last_reg = v_in # store this input for one cycle
        return v_out_valid, v_out_comp, v_out, inc_tb_ptr
