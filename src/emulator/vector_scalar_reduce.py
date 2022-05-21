import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# This block will reduce a vector to a scalar and pad with zeros
class VectorScalarReduce():
    def __init__(self,N):
        self.v_in=np.zeros(N)
        self.v_out=np.zeros(N)
        self.eof_in = [False,False]
        self.eof_out = [False,False]
        self.bof_in = [True,True]
        self.bof_out = [True,True]
        self.chainId_in = 0
        self.chainId_out = 0
        self.config=None
        self.N = N

    def step(self,input_value):
        # Reduce matrix along a given axis
        cfg=self.config[self.chainId_in]
        
        if cfg.op==0:
            log.debug('Passing first vector through vs reduce unit')
            self.v_out=self.v_in
        elif cfg.op==1:
            log.debug('Sum vector scalar reduce')
            self.v_out=np.concatenate(([np.sum(self.v_in)],np.zeros(self.N-1))) 
          
        self.eof_out, self.bof_out, self.chainId_out = self.eof_in, self.bof_in, self.chainId_in
        self.v_in, self.eof_in, self.bof_in, self.chainId_in = copy(input_value)
        return self.v_out, self.eof_out, self.bof_out, self.chainId_out

