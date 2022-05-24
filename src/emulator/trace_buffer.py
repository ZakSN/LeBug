import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# Circular Buffer
class TraceBuffer():
    def __init__(self,N,TB_SIZE,DELTA_SLOTS,PRECISION,INV):
        self.N = N
        self.TB_SIZE = TB_SIZE
        self.DELTA_SLOTS = DELTA_SLOTS
        self.PRECISION = PRECISION
        self.INV = INV

        # initial trace buffer state
        self.tbptr = self.TB_SIZE - 1 # trace buffer pointer
        nodata = n_bit_nodata(self.DELTA_SLOTS, self.PRECISION, self.INV)
        self.tbuffer = np.full((self.TB_SIZE, self.N), nodata) # trace buffer
        self.cfbuffer = np.full((self.TB_SIZE, 1), COMPRESSED) # compression flag buffer

    def step(self,compressed_data):
        v_out_valid, v_out_comp, v_out, inc_tb_ptr = compressed_data

        if not v_out_valid:
                return

        if inc_tb_ptr:
            self.tbptr = rollover_counter(self.tbptr, self.TB_SIZE)

        self.tbuffer[self.tbptr] = v_out
        if v_out_comp:
            self.cfbuffer[self.tbptr] = COMPRESSED
        else:
            self.cfbuffer[self.tbptr] = UNCOMPRESSED


