import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# Packs data efficiently
class DataPacker():
    def __init__(self,N,M,DATA_TYPE,DATA_WIDTH,DATA_MAX):
        self.v_in=np.zeros(N)
        self.v_out=np.zeros(N)
        self.eof_in = [False,False]
        self.bof_in = [True,True]
        self.chainId_in = 0
        self.v_out_valid=0
        self.v_out_size=0
        self.config=None
        self.N = N
        self.DATA_TYPE=DATA_TYPE
        self.DATA_WIDTH = DATA_WIDTH
        self.DATA_MAX = DATA_MAX

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
                log.debug('Data Packer full. Pushing values to Delta Compressor')
                self.v_out_valid=1
                self.v_out_size = 0
            else:
                self.v_out_valid=0
        else:
            self.v_out_valid=0
        self.v_in, self.eof_in, self.bof_in, self.chainId_in = copy(input_value)

        v_out_encoded = self.v_out
        if v_out_encoded.size == 1:
            # apparently numpy sometimes emits scalars, which are not in
            # an array
            v_out_encoded = np.reshape(v_out_encoded, v_out_encoded.size)
        # the output of numpy is always floats, but the delta compressor must
        # operate on either fixed point numbers or integers. Additionally, the
        # user may decide to do processing in fixed point arithmetic, but store
        # integer results (which are more compressible). Thus, we must also
        # enforce the cast_to_int firmware flag
        if cfg.cast_to_int or (self.DATA_TYPE == 'int'):
            # if the output should be an integer, cast it to int
            v_out_encoded = v_out_encoded.astype(int)
        else:
            # if we have a floating point number, turn it into a fixed point
            # number encoded in a two's complement integer
            def f(x):
                if x > self.DATA_MAX:
                    return x - ((2**self.DATA_WIDTH)-1)
                return x
            vect_f = np.vectorize(f)
            vector = np.squeeze(np.array(floatToEncodedInt(v_out_encoded,self.DATA_WIDTH)))
            v_out_encoded = vect_f(vector).astype(int)
        return v_out_encoded, self.v_out_valid

