import logging as log
import sys, math
import numpy as np
from firmware.compiler import compiler
from misc.misc import *

# import hardware block emulations
from emulator.input_buffer import InputBuffer
from emulator.filter_unit import FilterUnit
from emulator.matrix_vector_reduce import MatrixVectorReduce
from emulator.vector_scalar_reduce import VectorScalarReduce
from emulator.vector_vector_alu import VectorVectorALU
from emulator.data_packer import DataPacker
from emulator.trace_buffer import TraceBuffer

# Setting Debug level (can be debug, info, warning, error and critical)
log.basicConfig(stream=sys.stderr, level=log.INFO)

''' Emulation settings '''
DEBUG=True

class emulatedHw():
    def step(self):
        log.debug('New step')

        # Perform operations according to how building blocks are connected
        for b in self.BUILDING_BLOCKS:
            if b=='InputBuffer':
                chain = self.ib.step()
                self.log['ib'].append(chain)
            elif b=='FilterReduceUnit':
                chain = self.fu.step(chain)
                self.log['fu'].append(chain)
                chain = self.mvru.step(chain)
                self.log['mvru'].append(chain)
            elif b=='VectorVectorALU':
                chain = self.vvalu.step(chain)
                self.log['vvalu'].append(chain)
            elif b=='VectorScalarReduce':
                chain = self.vsru.step(chain)
                self.log['vsru'].append(chain)
            elif b=='DataPacker':
                packed_data = self.dp.step(chain)
                self.log['dp'].append(packed_data)
            elif b=='TraceBuffer':
                self.tb.step(packed_data)
                self.log['tb'].append(self.tb.mem)
            else:
                assert False, "Unknown building block "+b

    # Pushes values to the input of the chain
    def push(self,pushed_vals):
        self.ib.push(pushed_vals)

    def config(self,fw=None):
        # Configure processor
        # Fixme - For some reason I need to append a chain of zeros here
        no_cond={'last':False,'notlast':False,'first':False,'notfirst':False}
        self.fu.config=[struct(filter=0,addr=0)]
        self.mvru.config=[struct(axis=0)]
        self.vsru.config=[struct(op=0)]
        self.vvalu.config=[struct(op=0,addr=0,cache=0,cache_addr=0,cond1=copy(no_cond),cond2=copy(no_cond),minicache=0,cache_cond1=copy(no_cond),cache_cond2=copy(no_cond))]
        self.dp.config=[struct(commit=0,size=0,cond1=copy(no_cond),cond2=copy(no_cond))]
        self.ib.config=struct(num_chains=1)
        if fw is not None:
            self.ib.config=struct(num_chains=fw['valid_chains']+1)
            for idx in range(fw['valid_chains']):
                self.fu.config.append(fw['fu'][idx])
                self.mvru.config.append(fw['mvru'][idx])
                self.vsru.config.append(fw['vsru'][idx])
                self.vvalu.config.append(fw['vvalu'][idx])
                self.dp.config.append(fw['dp'][idx])

    def initialize_fu(vals):
        self.fu.vrf=vals

    def run(self,steps=50):
        # Keep stepping through the circuit as long as we have instructions to execute
        for i in range(steps):
            self.step()
        return self.log

    def __init__(self,N,M,IB_DEPTH,FUVRF_SIZE,VVVRF_SIZE,TB_SIZE,MAX_CHAINS,BUILDING_BLOCKS):
        ''' Verifying parameters '''
        assert math.log(N, 2).is_integer(), "N must be a power of 2" 
        assert math.log(M, 2).is_integer(), "N must be a power of 2" 
        assert M<=N, "M must be less or equal to N" 

        # hardware building blocks   
        self.BUILDING_BLOCKS=BUILDING_BLOCKS
        self.MAX_CHAINS=MAX_CHAINS
        self.ib   = InputBuffer(N,IB_DEPTH)
        self.fu   = FilterUnit(N,M,FUVRF_SIZE)
        self.mvru = MatrixVectorReduce(N,M)
        self.vsru = VectorScalarReduce(N)
        self.vvalu= VectorVectorALU(N,VVVRF_SIZE)
        self.dp   = DataPacker(N,M)
        self.tb   = TraceBuffer(N,TB_SIZE)
        self.config()

        # Firmware compiler
        self.compiler = compiler(N,M,MAX_CHAINS)

        # used to simulate a trace buffer to match results with simulation
        self.log={k: [] for k in ['ib','fu','mvru','vsru','vvalu','dp','tb']}
