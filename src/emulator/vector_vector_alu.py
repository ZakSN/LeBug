
import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# This block will reduce the matrix along a given axis
class VectorVectorALU():
    def __init__(self,N,VVVRF_SIZE):
        self.v_in=np.zeros(N)
        self.v_out=np.zeros(N)
        self.eof_in = [False,False]
        self.eof_out = [False,False]
        self.bof_in = [False,False]
        self.bof_out = [False,False]
        self.chainId_in = 0
        self.chainId_out = 0
        self.vrf=np.zeros(N*VVVRF_SIZE)
        self.config=None
        self.v_out_d1=np.zeros(N)
        self.v_out_d2=np.zeros(N)
        self.eof_out_d1 = [False,False]
        self.eof_out_d2 = [False,False]
        self.bof_out_d1 = [True,True]
        self.bof_out_d2 = [True,True]
        self.chainId_out_d2 = 0
        self.chainId_out_d1 = 0
        self.N = N
        self.minicache = np.zeros(N)

    def step(self,input_value):
        # Delay for 2 cycles, so this FU takes 3 cycles (read, calculate, write)
        self.v_out    = self.v_out_d2
        self.v_out_d2 = self.v_out_d1
        self.eof_out  = self.eof_out_d2
        self.eof_out_d2  = self.eof_out_d1
        self.eof_out_d1  = self.eof_in
        self.bof_out  = self.bof_out_d2
        self.bof_out_d2  = self.bof_out_d1
        self.bof_out_d1  = self.bof_in
        self.chainId_out  = self.chainId_out_d2
        self.chainId_out_d2  = self.chainId_out_d1
        self.chainId_out_d1  = self.chainId_in
        cfg=self.config[self.chainId_in]
        condition_met = ((not cfg.cond1['last']     or (cfg.cond1['last']     and     self.eof_in[0])) and
                         (not cfg.cond1['notlast']  or (cfg.cond1['notlast']  and not self.eof_in[0])) and
                         (not cfg.cond1['first']    or (cfg.cond1['first']    and     self.bof_in[0])) and
                         (not cfg.cond1['notfirst'] or (cfg.cond1['notfirst'] and not self.bof_in[0])) and
                         (not cfg.cond2['last']     or (cfg.cond2['last']     and     self.eof_in[1])) and
                         (not cfg.cond2['notlast']  or (cfg.cond2['notlast']  and not self.eof_in[1])) and
                         (not cfg.cond2['first']    or (cfg.cond2['first']    and     self.bof_in[1])) and
                         (not cfg.cond2['notfirst'] or (cfg.cond2['notfirst'] and not self.bof_in[1])))

        # Checking if we should use minicache or input vector as operator
        if cfg.minicache == 1 or cfg.minicache==3:
            operator = self.minicache
        else:
            operator = self.v_in

        if cfg.op==0 or not condition_met:
            log.debug('ALU is passing values through')
            self.v_out_d1 = operator
        elif cfg.op==1:
            log.debug('Adding using vector-vector ALU')
            self.v_out_d1 = operator + self.vrf[cfg.addr*self.N:cfg.addr*self.N+self.N]
        elif cfg.op==2:
            log.debug('Multiplying using vector-vector ALU')
            self.v_out_d1 = operator * self.vrf[cfg.addr*self.N:cfg.addr*self.N+self.N]
        elif cfg.op==3:
            log.debug('Subtracting using vector-vector ALU')
            self.v_out_d1 = operator - self.vrf[cfg.addr*self.N:cfg.addr*self.N+self.N]
        elif cfg.op==4:
            log.debug('Subtracting using vector-vector ALU')
            self.v_out_d1 = np.maximum(operator, self.vrf[cfg.addr*self.N:cfg.addr*self.N+self.N])

        cache_condition_met = ((not cfg.cache_cond1['last']     or (cfg.cache_cond1['last']     and     self.eof_in[0])) and
                         (not cfg.cache_cond1['notlast']  or (cfg.cache_cond1['notlast']  and not self.eof_in[0])) and
                         (not cfg.cache_cond1['first']    or (cfg.cache_cond1['first']    and     self.bof_in[0])) and
                         (not cfg.cache_cond1['notfirst'] or (cfg.cache_cond1['notfirst'] and not self.bof_in[0])) and
                         (not cfg.cache_cond2['last']     or (cfg.cache_cond2['last']     and     self.eof_in[1])) and
                         (not cfg.cache_cond2['notlast']  or (cfg.cache_cond2['notlast']  and not self.eof_in[1])) and
                         (not cfg.cache_cond2['first']    or (cfg.cache_cond2['first']    and     self.bof_in[1])) and
                         (not cfg.cache_cond2['notfirst'] or (cfg.cache_cond2['notfirst'] and not self.bof_in[1])))
        if cfg.cache & cache_condition_met:
            self.vrf[cfg.cache_addr*self.N:cfg.cache_addr*self.N+self.N] = self.v_out_d1 
        if cfg.minicache==2 or cfg.minicache==3:
            self.minicache = copy(self.v_out_d1) 
        
        self.v_in, self.eof_in, self.bof_in, self.chainId_in = copy(input_value)
        return self.v_out, self.eof_out, self.bof_out, self.chainId_out

