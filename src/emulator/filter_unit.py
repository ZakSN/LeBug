import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# Filter Unit
class FilterUnit():
    def __init__(self,N,M,FUVRF_SIZE):
        self.v_in=np.zeros(N)
        self.m_out=np.zeros((M,N))
        self.eof_in = [False,False]
        self.eof_out = [False,False]
        self.bof_in = [True,True]
        self.bof_out = [True,True]
        self.chainId_in = 0
        self.chainId_out = 0
        self.vrf=np.zeros(FUVRF_SIZE*M)
        self.config=None
        self.M = M
        self.N = N

    def step(self,input_value):
        # Check if the vector is within M ranges
        cfg=self.config[self.chainId_in]
        log.debug('Filter input:'+str(self.v_in))
        log.debug('Filtering using the following ranges:'+str(self.vrf[cfg.addr*self.M:cfg.addr*self.M+self.M+1]))
        if cfg.filter==1:
            for i in range(self.M):
                low_range = self.vrf[cfg.addr*self.M+i]
                if cfg.addr*self.M+i+1<len(self.vrf):
                    high_range = self.vrf[cfg.addr*self.M+i+1]
                else:
                    high_range = low_range+(low_range-self.vrf[cfg.addr*self.M+i-1])
                within_range = np.all([self.v_in>low_range, self.v_in<=high_range],axis=0)
                self.m_out[i]=within_range[:]
        # If we are not filtering, just pass the value through 
        else:
            for i in range(self.M):
                self.m_out[i] = self.v_in if i==0 else np.zeros(self.N)

        self.eof_out, self.bof_out, self.chainId_out = self.eof_in, self.bof_in, self.chainId_in
        self.v_in, self.eof_in, self.bof_in, self.chainId_in = copy(input_value)
        return self.m_out, self.eof_out, self.bof_out, self.chainId_out

