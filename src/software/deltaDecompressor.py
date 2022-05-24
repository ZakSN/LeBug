import numpy as np
from misc.misc import *

class deltaDecompressor():
    def __init__(self,N,DATA_WIDTH,DELTA_SLOTS,TB_SIZE):
        self.N = N
        assert DATA_WIDTH%DELTA_SLOTS == 0, "data width must be divisible by delta slots"
        self.PRECISION = int(DATA_WIDTH/DELTA_SLOTS)
        self.DELTA_SLOTS = DELTA_SLOTS
        self.TB_SIZE = TB_SIZE
        self.INV = twos_complement_min(self.PRECISION)

    def decompress(self, last_reg, tbuffer, cfbuffer, tbptr):
        def unpack_deltas(pvector):
            deltas = np.full(self.N, None)
            mask = np.full(self.N, -1, dtype=int)
            mask = mask << int(self.PRECISION)
            mask = np.bitwise_not(mask)
            for i in range(self.DELTA_SLOTS):
                delta = np.bitwise_and(pvector, mask)
                delta = sign_extend(delta, self.PRECISION)
                if np.all(np.full(self.N, self.INV, dtype=int) != delta):
                    deltas = np.vstack([deltas, delta])
                pvector = pvector >> int(self.PRECISION)
            deltas = deltas[1:]
            return deltas
        stop = tbptr
        decompressed = np.array([last_reg])

        # The following ought to be a do-while, but python doesn't have
        # this structure
        while True:
            if cfbuffer[tbptr] == COMPRESSED:
                deltas = unpack_deltas(tbuffer[tbptr])
                if not np.all(deltas == None):
                    for i in range(np.shape(deltas)[0]):
                        last_reg = last_reg + deltas[i]
                        decompressed = np.vstack([last_reg, decompressed])
            else:
                last_reg = tbuffer[tbptr]
                decompressed = np.vstack([last_reg, decompressed])

            tbptr = rollover_counter(tbptr, self.TB_SIZE, False)

            if tbptr == stop:
                break

        return decompressed.astype(int)
