import sys
sys.path.insert(1, '../../src/')
from emulator.emulator import emulatedHw
from hardware.hardware import rtlHw
from software.delta_decompressor import DeltaDecompressor
from misc.misc import *
import firmware.firmware as firm
import math
import numpy as np
np.set_printoptions(precision=3, suppress=False)
import unittest
import warnings

class TestHardware(unittest.TestCase):
    # Read YAML configuration file and declare those as global variables
    def setUp(self):
        # FIXME: silence warnings, so that we don't get overwhelmed with warnings
        # from the (kinda buggy) docker python SDK
        # specifically version check deprecation warnings, and resource warnings
        # about unclosed sockets:
        # https://github.com/docker/docker-py/issues/1293
        warnings.simplefilter('ignore', category=DeprecationWarning)
        warnings.simplefilter('ignore', category=ResourceWarning)

        with open(r'config.yaml') as file:
            yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.cfg = yaml_dict
        self.N = yaml_dict['N']
        self.M = yaml_dict['M']
        self.IB_DEPTH = yaml_dict['IB_DEPTH']
        self.FUVRF_SIZE = yaml_dict['FUVRF_SIZE']
        self.VVVRF_SIZE = yaml_dict['VVVRF_SIZE']
        self.TB_SIZE = yaml_dict['TB_SIZE']
        self.MAX_CHAINS = yaml_dict['MAX_CHAINS']
        self.BUILDING_BLOCKS = yaml_dict['BUILDING_BLOCKS']
        self.DATA_WIDTH = yaml_dict['DATA_WIDTH']
        self.DELTA_SLOTS = yaml_dict['DELTA_SLOTS']
        self.DATA_TYPE = yaml_dict['DATA_TYPE']
        self.DEVICE_FAM = yaml_dict['DEVICE_FAM']

    # Filter results after computations
    def filterResults(self, emu_results, hw_results, DATA_TYPE):
        dd = DeltaDecompressor(
            self.N,
            self.DATA_WIDTH,
            self.DELTA_SLOTS,
            self.TB_SIZE
        )
        emu_results_filtered = dd.decompress(
            emu_results['dc'][-1][1],
            emu_results['tb'][-1][0],
            emu_results['tb'][-1][1],
            emu_results['tb'][-1][2],
        )
        if DATA_TYPE=='int':
            hw_results_filtered = np.array(toInt(hw_results['tb']['mem_data']))
        elif DATA_TYPE=='fixed_point':
            hw_results_filtered = np.array(encodedIntTofloat(hw_results['tb']['mem_data'],self.DATA_WIDTH))

        # Print Results
        print("\n\n********** Emulation results **********")
        print(emu_results_filtered)
        print("\n********** Hardware results **********")
        print(hw_results_filtered)

        return emu_results_filtered, hw_results_filtered

    # Put input values into testbench
    def pushVals(self, emu_proc,hw_proc,num_input_vectors,eof1=None,eof2=None,neg_vals=False):
        np.random.seed(0)
        input_vectors=[]
        if eof1 is None:
            eof1=num_input_vectors*[False]
        if eof2 is None:
            eof2=num_input_vectors*[False]
        print("********** Input vectors **********")
        for i in range(num_input_vectors):
            # Integer data type
            if hw_proc.DATA_TYPE==0:
                input_vectors.append(np.random.randint(9, size=self.N))
                print(f'Cycle {i}:\t{input_vectors[i]}')
                emu_proc.push([input_vectors[i],eof1[i],eof2[i]])
            # Fixed-point data type
            elif hw_proc.DATA_TYPE==1:
                if neg_vals:
                    input_vectors.append(10*np.random.random(self.N)-5)
                else:
                    input_vectors.append(10*np.random.random(self.N))
                print(f'Cycle {i}:\t{input_vectors[i]}')
                emu_proc.push([input_vectors[i],eof1[i],eof2[i]])
                input_vectors[i] = floatToEncodedInt(input_vectors[i],hw_proc.DATA_WIDTH)
            hw_proc.push([input_vectors[i],eof1[i],eof2[i]])

    def test_raw(self):

        # Instantiate HW and Emulator Processors
        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=3
        self.pushVals(emu_proc,hw_proc,num_input_vectors,neg_vals=True)

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.raw(hw_proc.compiler)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=30
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.DATA_TYPE)

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered,rtol=0.01))

    def test_multipleChains(self):

        # Instantiate HW and Emulator Processors
        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=1
        self.pushVals(emu_proc,hw_proc,num_input_vectors,neg_vals=True)

        # Initialize the memories the same way
        emu_proc.initialize_fu=list(range(self.FUVRF_SIZE*self.M))
        hw_proc.initialize_fu=list(range(self.FUVRF_SIZE*self.M))

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.multipleChains(hw_proc.compiler)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=30
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.DATA_TYPE)

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered, rtol=0.01))

    def test_correlation(self):
        # store parameter_state
        old_parameter_state = self.cfg

        # Instantiate HW and Emulator Processors
        self.cfg['TB_SIZE']=10
        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=5
        self.pushVals(emu_proc,hw_proc,num_input_vectors,neg_vals=True)

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.correlation(hw_proc.compiler)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=30
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.DATA_TYPE)

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered, rtol=0.05))

        # restore parameter_state
        self.cfg = old_parameter_state

    def test_conditions(self):

        # Instantiate HW and Emulator Processors
        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=5
        eof1=[False,False,True,False,True]
        self.pushVals(emu_proc,hw_proc,num_input_vectors,eof1)

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.conditions(hw_proc.compiler)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=45
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.DATA_TYPE)

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered, rtol=0.05))

    def test_distribution(self):

        # Instantiate HW and Emulator Processors
        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=2
        eof1=num_input_vectors*[True]
        self.pushVals(emu_proc,hw_proc,num_input_vectors,eof1)

        # Initialize the memories the same way
        emu_proc.initialize_fu=list(range(self.FUVRF_SIZE*self.M))
        hw_proc.initialize_fu=list(range(self.FUVRF_SIZE*self.M))

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.distribution(hw_proc.compiler,16,4)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=45
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.DATA_TYPE)

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered))

    def test_minicache_test(self):
        # store parameter_state
        old_parameter_state = self.cfg

        # Instantiate HW and Emulator Processors
        self.cfg['DATA_TYPE']='int'
        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=3
        eof1=num_input_vectors*[True]
        self.pushVals(emu_proc,hw_proc,num_input_vectors,eof1)

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.minicache(hw_proc.compiler)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=45
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.cfg['DATA_TYPE'])

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered,rtol=0.05))

        # restore parameter_state
        self.cfg = old_parameter_state

    def test_predictiveness(self):
        # store parameter_state
        old_parameter_state = self.cfg

        # Instantiate HW and Emulator Processors
        self.cfg['DATA_TYPE']='int'
        self.cfg['BUILDING_BLOCKS']=['InputBuffer', 'FilterReduceUnit','VectorScalarReduce','VectorVectorALU','DataPacker','TraceBuffer']

        hw_proc  = rtlHw(**self.cfg)
        emu_proc = emulatedHw(**self.cfg)

        # Create common input values
        num_input_vectors=4
        eof1=[False,True,False,True]
        eof2=[False,False,False,True]
        self.pushVals(emu_proc,hw_proc,num_input_vectors,eof1,eof2)

        # Configure firmware - Both HW and Emulator work with the same firmware
        fw = firm.activationPredictiveness(hw_proc.compiler)
        emu_proc.config(fw)
        hw_proc.config(fw)

        # Run HW simulation and emulation
        steps=45
        hw_results = hw_proc.run(steps=steps,gui=False,log=False)
        emu_results = emu_proc.run(steps=steps)

        # Filter Results
        emu_results_filtered, hw_results_filtered = self.filterResults(emu_results, hw_results, self.cfg['DATA_TYPE'])

        # Verify that results are equal
        self.assertTrue(np.allclose(emu_results_filtered,hw_results_filtered,rtol=0.05))

        # restore parameter_state
        self.cfg = old_parameter_state

if __name__=="__main__":
    unittest.main()
