from copy import deepcopy as copy
import yaml
import numpy as np

''' C-like struct '''
class struct:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
    def __repr__(self):
        return str(self.__dict__)

''' Map list to int '''
def toInt(lst):
    return [list(map(int, l)) for l in lst]

''' Encode vector of floats to ints '''
def floatToEncodedInt(float_array,DATA_WIDTH):
    return [encode(x,DATA_WIDTH) for x in float_array]

''' Encode vector of floats to ints '''    
def encode(value,DATA_WIDTH):
    int_bits=int(DATA_WIDTH/2)
    frac_bits=int(DATA_WIDTH/2)
    is_negative = value<0
    max_value = (1<<(int_bits-1+frac_bits))-1
    x = round(value * (1<< frac_bits))
    x = int(max_value if x > max_value else -max_value if x< -max_value else x)
    if is_negative:
        x = (1<<DATA_WIDTH) + x
    return x

''' Decode vector of floats from encoded ints back to floats '''
def encodedIntTofloat(encoded_int,DATA_WIDTH):
    frac_bits=int(DATA_WIDTH/2)
    return [[decode(encoded_value,DATA_WIDTH) for encoded_value in l] for l in encoded_int] 

''' Decode vector of floats from encoded ints back to floats '''
def decode(value,DATA_WIDTH):
    int_bits=int(DATA_WIDTH/2)
    frac_bits=int(DATA_WIDTH/2)
    value=float(value)
    max_value = (1<<(int_bits-1+frac_bits))-1
    is_negative = value>max_value
    if is_negative:
        value = -((1<<DATA_WIDTH) - value)
    return value / (1 << frac_bits)

''' Maximum two's complement number that fits in n bits'''
def twos_complement_max(n):
    return int((2**(n-1)) - 1)

''' Minimum two's complement number that fits in n bits'''
def twos_complement_min(n):
    return int(-1*(2**(n-1)))

''' Two's complement sign extension, vectorized from: 
    https://stackoverflow.com/questions/38803320/sign-extending-from-a-variable-bit-width '''
def sign_extend(x, b):
    return np.piecewise(x, x&(1<<(b-1)),
        [lambda z: z-(1<<b), lambda z: z])

''' Check that a numpy vector is the correct shape and has elements in the
    correct ranges '''
def assert_vector_size(vector, num_elements, element_min, element_max):
    assert len(vector) == num_elements, "vector wrong size"
    assert np.all(vector>=element_min), str(vector)+" has element smaller than: "+str(element_min)
    assert np.all(vector<=element_max), str(vector)+" has element larger than: "+str(element_max)

''' Simple rollover counter, increment d between 0 and r, unless direction is
    set in which case decrement d'''
def rollover_counter(d, r, direction=None):
    if direction == None:
        d = d + 1
    else:
        d = d - 1
    if (0 <= d) and (d < r):
        return d
    elif direction == None:
        return 0
    else:
        return r - 1

''' Compression flag buffer compressed/uncompressed flag constants '''
UNCOMPRESSED = 1
COMPRESSED = 0

''' Create an n bit no data symbol '''
def n_bit_nodata(DELTA_SLOTS, PRECISION, INV):
    nodata = 0
    for i in range(DELTA_SLOTS):
        nodata = nodata << PRECISION
        nodata = nodata | (INV & (2**PRECISION-1))
    return nodata
