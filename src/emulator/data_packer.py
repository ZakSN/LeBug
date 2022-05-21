import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# Packs data efficiently
class DataPacker():
    def __init__(self,N,M):
        self.v_in=np.zeros(N)
        self.v_out=np.zeros(N)
        self.eof_in = [False,False]
        self.bof_in = [True,True]
        self.chainId_in = 0
        self.v_out_valid=0
        self.v_out_size=0
        self.config=None
        self.N = N

    def step(self,input_value):
        cfg=self.config[self.chainId_in]
        if (cfg.commit and 
            (not cfg.cond1['last']     or (cfg.cond1['last']     and     self.eof_in[0])) and
            (not cfg.cond1['notlast']  or (cfg.cond1['notlast']  and not self.eof_in[0])) and
            (not cfg.cond1['first']    or (cfg.cond1['first']    and     self.bof_in[0])) and
            (not cfg.cond1['notfirst'] or (cfg.cond1['notfirst'] and not self.bof_in[0])) and
            (not cfg.cond2['last']     or (cfg.cond2['last']     and     self.eof_in[1])) and
            (not cfg.cond2['notlast']  or (cfg.cond2['notlast']  and not self.eof_in[1])) and
            (not cfg.cond2['first']    or (cfg.cond2['first']    and     self.bof_in[1])) and
            (not cfg.cond2['notfirst'] or (cfg.cond2['notfirst'] and not self.bof_in[1]))):
            if self.v_out_size==0:
                self.v_out = self.v_in[:cfg.size]
            else:
                self.v_out = np.append(self.v_out,self.v_in[:cfg.size])
            self.v_out_size=self.v_out_size+cfg.size
            if self.v_out_size==self.N:
                log.debug('Data Packer full. Pushing values to Trace Buffer')
                self.v_out_valid=1
                self.v_out_size = 0
            else:
                self.v_out_valid=0
        else:
            self.v_out_valid=0
        self.v_in, self.eof_in, self.bof_in, self.chainId_in = copy(input_value)
        return self.v_out, self.v_out_valid

