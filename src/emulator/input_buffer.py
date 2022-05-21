import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# Input buffer class 
class InputBuffer():
    def __init__(self,N,IB_DEPTH):
        self.buffer=[]
        self.N = N
        self.size=IB_DEPTH
        self.config=None
        self.chainId_out = 0
        self.bof_out=[True,True]

    def push(self,pushed_vals):
        eof_in = [False,False]
        if len(pushed_vals)==2:
            pushed_vals.append(False)
        v_in, eof_in[0], eof_in[1] = pushed_vals
        assert list(v_in.shape)==[self.N], "Input must be Nx1"
        assert len(self.buffer)<=self.size, "Input buffer overflowed"
        log.debug('Vector inserted into input buffer\n'+str(v_in))
        self.buffer.append([v_in,eof_in])

    def pop(self):
        log.debug("Removing element from input buffer")
        assert len(self.buffer)>0, "Input buffer is empty"
        self.bof_out=self.buffer[0][1]
        return self.buffer.pop(0)

    def step(self):
        # Dispatch a new chain if the input buffer is not empty
        # Note that if our FW has 3 chains num_chains will be 4, since we need one "chain" (chainId 0) to work as a pass through
        if len(self.buffer)>0:
            if self.chainId_out<self.config.num_chains:
                # Go to next element in the input buffer once we dispatched all chains for the previous element
                if self.chainId_out==self.config.num_chains-1:
                    self.pop()
                    self.chainId_out = 0 if len(self.buffer)==0 else 1 
                else:
                    self.chainId_out=self.chainId_out+1

        # If the trace buffer is full, we will dispatch chain 0, which is a pass through
        else:
            self.chainId_out=0

        if len(self.buffer)>0:
            v_out, eof_out = self.buffer[0]
        else:
            v_out, eof_out = np.zeros(self.N), False
        return v_out, eof_out, self.bof_out, self.chainId_out

