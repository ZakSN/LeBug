import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# This block will reduce the matrix along a given axis
# If M<N, then the results will be padded with zeros
class MatrixVectorReduce():
    def __init__(self,N,M):
        self.m_in=np.zeros((M,N))
        self.v_out=np.zeros(N)
        self.eof_in = [False,False]
        self.eof_out = [False,False]
        self.bof_in = [True,True]
        self.bof_out = [True,True]
        self.chainId_in = 0
        self.chainId_out = 0
        self.config=None
        self.N = N
        self.M = M

    def step(self,input_value):
        # Reduce matrix along a given axis
        cfg=self.config[self.chainId_in]
        if cfg.axis==0:
            log.debug('Passing first vector through reduce unit')
            self.v_out=self.m_in[0]
        elif cfg.axis==1:
            log.debug('Reducing matrix along N axis (axis = '+str(cfg.axis)+')')
            self.v_out=np.sum(self.m_in,axis=0)
        elif cfg.axis==2:
            log.debug('Reducing matrix along M axis (axis = '+str(cfg.axis)+')')
            self.v_out=np.sum(self.m_in,axis=1)
            if self.N!=self.M:
                log.debug('Padding results with '+str(self.N-self.M)+' zeros')
                self.v_out=np.concatenate((self.v_out,np.zeros(self.N-self.M)))

        self.eof_out, self.bof_out, self.chainId_out    = self.eof_in, self.bof_in, self.chainId_in
        self.m_in, self.eof_in, self.bof_in, self.chainId_in = copy(input_value)
        return self.v_out, self.eof_out, self.bof_out, self.chainId_out

