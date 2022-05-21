import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# Packs data efficiently
class TraceBuffer():
    def __init__(self,N,TB_SIZE):
        self.input=np.zeros(N)
        self.mem=np.zeros((TB_SIZE,N))
        self.size=0
        self.TB_SIZE=TB_SIZE

    def step(self,packed_data):
        output, output_valid = packed_data
        if output_valid:
            if self.size==self.TB_SIZE:
                self.size=0
            self.mem[self.size]=output
            self.size=self.size+1
        self.input=copy(packed_data)

