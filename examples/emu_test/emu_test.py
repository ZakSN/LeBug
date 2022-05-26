import sys
sys.path.insert(1, '../../src/')
from emulator.emulator import emulatedHw
from hardware.hardware import rtlHw
from software.delta_decompressor import DeltaDecompressor
import firmware.firmware as firm
import math, yaml
import numpy as np
import unittest

class TestEmulator(unittest.TestCase):
    # Read YAML configuration file and declare those as global variables
    def setUp(self):
        with open(r'config.yaml') as file:
            yaml_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.emu_cfg = yaml_dict
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

    def test_SimpleDistribution(self):
        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)

        # Initial hardware setup
        proc.fu.vrf=list(range(self.FUVRF_SIZE*self.M)) # Initializing fuvrf

        fw = firm.distribution(proc.compiler,bins=2*self.M,M=self.M)

        # Feed one value to input buffer
        np.random.seed(42)
        input_vector = np.random.rand(self.N)*8
        proc.push([input_vector,True])

        # Step through it until we get the result
        proc.config(fw)
        log = proc.run()

        # Decompress the tracebuffer
        dd = DeltaDecompressor(self.N,self.DATA_WIDTH,self.DELTA_SLOTS,self.TB_SIZE)
        decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])

        # Test distribution
        self.assertTrue(np.allclose(decomp_tb[-1],[ 1,2,1,0,1,1,1,1]))

    def test_DualDistribution(self):

        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)

        # Initial hardware setup
        proc.fu.vrf=list(range(self.FUVRF_SIZE*self.M)) # Initializing fuvrf

        fw = firm.distribution(proc.compiler,bins=2*self.M,M=self.M)

        # Feed one value to input buffer
        np.random.seed(42)
        input_vector1=np.random.rand(self.N)*8
        input_vector2=np.random.rand(self.N)*8

        proc.push([input_vector1,False])
        proc.push([input_vector2,True])

        # Step through it until we get the result
        proc.config(fw)
        log = proc.run()

        # Decompress the tracebuffer
        dd = DeltaDecompressor(self.N,self.DATA_WIDTH,self.DELTA_SLOTS,self.TB_SIZE)
        decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])

        # Test dual distribution
        self.assertTrue(np.allclose(decomp_tb[-1],[ 2.,5.,1.,0.,2.,2.,2.,2.]))

    def test_SummaryStats(self):

        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)

        proc.fu.vrf=list(np.concatenate(([0.,float('inf')],list(reversed(range(self.FUVRF_SIZE*self.M-2)))))) # Initializing fuvrf for sparsity
        fw = firm.summaryStats(proc.compiler)

        # Feed one value to input buffer
        np.random.seed(0)
        input_vector1=np.random.rand(self.N)*8-4
        input_vector2=np.random.rand(self.N)*8-4

        proc.push([input_vector1,False])
        proc.push([input_vector1,False])
        proc.push([input_vector1,False])
        proc.push([input_vector1,False])
        proc.push([input_vector1,False])
        proc.push([input_vector1,False])
        proc.push([input_vector1,False])
        proc.push([input_vector2,True])

        # Step through it until we get the result
        proc.config(fw)
        log = proc.run()

        # Test reduce sum
        self.assertTrue(np.isclose(proc.dp.v_out[0],np.sum(input_vector1)*7+np.sum(input_vector2)))
        # Test Sparsity sum
        self.assertTrue(47==int(proc.dp.v_out[1]))

    def test_SpatialSparsity(self):

        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)
        proc.fu.vrf=list(np.concatenate(([0.,float('inf')],list(reversed(range(self.FUVRF_SIZE*self.M-2)))))) # Initializing fuvrf for sparsity
        fw = firm.spatialSparsity(proc.compiler,self.N)

        # Feed one value to input buffer
        np.random.seed(0)
        input_vector1=np.random.rand(self.N)*8-4
        input_vector2=np.random.rand(self.N)*8-4

        proc.push([input_vector1,False])
        proc.push([input_vector2,True])

        # Step through it until we get the result
        proc.config(fw)
        log = proc.run()

        # Decompress the tracebuffer
        dd = DeltaDecompressor(self.N,self.DATA_WIDTH,self.DELTA_SLOTS,self.TB_SIZE)
        decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])

        # Test spatial sparsity
        self.assertTrue(np.allclose(decomp_tb[0],[ 1.,1.,1.,1.,0.,1.,0.,1.]))
        self.assertTrue(np.allclose(decomp_tb[1],[ 1.,0.,1.,1.,1.,1.,0.,0.]))

    def test_Correlation(self):

        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)
        fw = firm.correlation(proc.compiler)

        # Feed one value to input buffer
        np.random.seed(0)
        input_vector1=np.random.rand(self.N)*8-4
        input_vector2=np.random.rand(self.N)*8-4

        proc.push([input_vector1,False])
        proc.push([input_vector2,True])

        # Step through it until we get the result
        proc.config(fw)
        tb = proc.run()

        # Note that this is the Cross-correlation of two 1-dimensional sequences, not the coeficient
        numpy_correlate = np.corrcoef(input_vector1,input_vector2)

        v = proc.dp.v_out
        x, y, xx, yy, xy = v[1], v[4], v[2], v[5], v[3]

        # Note that the equation in this website is wrong, but the math is correct
        # https://www.investopedia.com/terms/c/correlation.asp

        corr = (self.N*xy-x*y)/math.sqrt((self.N*xx-x*x)*(self.N*yy-y*y))

        # Test Correlation
        self.assertTrue(np.isclose(numpy_correlate[0][1],corr))

    def test_VectorChange(self):

        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)
        fw = firm.vectorChange(proc.compiler)

        # Feed value to input buffer
        np.random.seed(0)
        input_vector1=np.random.rand(self.N)*8-4
        input_vector2=np.random.rand(self.N)*8-4

        proc.push([input_vector1,False])
        proc.push([input_vector2,True])

        # Step through it until we get the result
        proc.config(fw)
        tb = proc.run()

        # Test vector change
        self.assertTrue(np.isclose(np.sum(input_vector2-input_vector1),proc.dp.v_out[1]))

    def test_DeltaCompression(self):

        # Instantiate processor
        proc = emulatedHw(**self.emu_cfg)
        fw = firm.raw(proc.compiler)

        # some values to compress/decompress
        frame_in = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 8, 10, 110000],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 8, 10, 110000],
            [5, 6, 7, 8, 9, 10, 11, 110001],
            [6, 7, 8, 9, 10, 11, 12, 110002],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ])

        proc.config(fw)

        # push the frame in
        for f in range(frame_in.shape[0]-1):
            proc.push([frame_in[f],False])

        # push the last value with EOF set
        proc.push([frame_in[-1],True])

        # run the pipeline
        log = proc.run()

        # Decompress the tracebuffer
        dd = DeltaDecompressor(self.N,self.DATA_WIDTH,self.DELTA_SLOTS,self.TB_SIZE)
        decomp_tb = dd.decompress(log['dc'][-1][1], log['tb'][-1][0], log['tb'][-1][1], log['tb'][-1][2])

        # should get the same out, except truncated since the TB has rolled over
        frame_out = np.array([
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 8, 10, 110000],
            [5, 6, 7, 8, 9, 10, 11, 12],
            [6, 7, 8, 9, 10, 11, 12, 13],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 3, 4, 5, 6, 7, 8, 9],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [4, 5, 6, 7, 8, 8, 10, 110000],
            [5, 6, 7, 8, 9, 10, 11, 110001],
            [6, 7, 8, 9, 10, 11, 12, 110002],
            [7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15],
        ])

        # Test raw compression
        self.assertTrue(np.all(decomp_tb == frame_out))

if __name__ == "__main__":
    unittest.main()
