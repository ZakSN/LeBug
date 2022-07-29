import numpy as np
import os
import sys
sys.path.insert(1, '../../../src')
from emulator.emulator import emulatedHw
import firmware.firmware as firm
sys.path.insert(1, '../../../examples')
from test_utils import TestUtils
from misc.misc import *
import math
import multiprocessing
import pickle

# class to encapsulate compression experiments
class AbstractCompressionExperiment():
    def __init__(self):
        # default emulated hardware configuration
        self.emu_cfg = {
            'N':               32,
            'M':               32,
            'IB_DEPTH':        32,
            'FUVRF_SIZE':      4,
            'VVVRF_SIZE':      8,
            'TB_SIZE':         64,
            'MAX_CHAINS':      8,
            'DATA_WIDTH':      32,
            'DELTA_SLOTS':     4,
            'DATA_TYPE':       'fixed_point',
            'BUILDING_BLOCKS': ['InputBuffer',
                                'FilterReduceUnit',
                                'VectorVectorALU',
                                'VectorScalarReduce',
                                'DataPacker',
                                'DeltaCompressor',
                                'TraceBuffer'],
        }
        self.emu_cfg['PRECISION'] = int(self.emu_cfg['DATA_WIDTH']/self.emu_cfg['DELTA_SLOTS'])
        self.emu_cfg['INV'] = twos_complement_min(self.emu_cfg['PRECISION'])

        # default end of frame bit configuration
        self.eof1_cfg = (
            False, # initial value
            True, # final value
            lambda i: i # update function
        )
        self.eof2_cfg = (
            False, # initial value
            False, # final value
            lambda i: i
        )

    #XXX need raw int and raw fxp?
    def raw(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        fw = firm.raw(emu.compiler)
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def distribution(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']))
        # can massage distribution configuration to effect compressibility
        fw = firm.distribution(emu.compiler,bins=self.emu_cfg['M'],M=self.emu_cfg['M'])
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    # computes combined summary: 
    # sum of all elements (incompressible, unless integers)
    # and sparsity count (compressible)
    # when combined the results are not compressible
    def summary(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(np.concatenate((
            [0.,float('inf')],
            list(reversed(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']-2))))))
        fw = firm.summaryStats(emu.compiler)
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def sparsity_count(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(np.concatenate((
            [0.,float('inf')],
            list(reversed(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']-2))))))
        fw = firm.numSparse(emu.compiler)
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def spatial_sparsity(self, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        emu.fu.vrf=list(np.concatenate((
            [0.,float('inf')],
            list(reversed(range(self.emu_cfg['FUVRF_SIZE']*self.emu_cfg['M']-2))))))
        fw = firm.spatialSparsity(emu.compiler, self.emu_cfg['N'])
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    def norm_check(self, limits=None, **kwargs):
        emu = emulatedHw(**self.emu_cfg)
        # XXX how to set fru filter values appropriately?
        emu.fu.vrf = list(np.linspace(limits[0], limits[1], 64))
        fw = firm.normCheck(emu.compiler, self.emu_cfg['M'])
        emu.config(fw)
        ret = {
            'emu_cfg' : self.emu_cfg,
            'emu' : emu,
            'eof1_cfg' : self.eof1_cfg,
            'eof2_cfg' : self.eof2_cfg,
        }
        return ret

    # I think this is a plausible control signal configuration, however in
    # practice the specific control signal manipulation may depend on how the
    # tensors are flattened
    def activation_predictiveness(self, **kwargs):
        this_emu_cfg = self.emu_cfg
        this_emu_cfg['BUILDING_BLOCKS'] = [
            'InputBuffer',
            'FilterReduceUnit',
            'VectorScalarReduce',
            'VectorVectorALU',
            'DataPacker',
            'DeltaCompressor',
            'TraceBuffer']
        emu = emulatedHw(**this_emu_cfg)
        fw = firm.activationPredictiveness(emu.compiler)
        emu.config(fw)
        this_eof1_cfg = (self.eof1_cfg[0], self.eof1_cfg[1], lambda i: ~i)
        this_eof2_cfg = (True, False, lambda i: False)
        ret = {
            'emu_cfg' : this_emu_cfg,
            'emu' : emu,
            'eof1_cfg' : this_eof1_cfg,
            'eof2_cfg' : this_eof2_cfg,
        }
        return ret

    # missing total invalidity
    def run_experiment(*args, **kwargs):
        raise NotImplementedError

