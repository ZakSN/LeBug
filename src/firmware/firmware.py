# This files contains some of the different firmware that can be used by the HW and emulator

# Firmware for a distribution with multiple sets of N values
def distribution(cp,bins,M):
    assert bins%M==0, "Number of bins must be divisible by M for now"
    for i in range(int(bins/M)):
        cp.begin_chain()
        cp.vv_filter(i)
        cp.m_reduce('M')
        cp.vv_add(i,'notfirst')
        cp.v_cache(i)
        cp.v_commit(M,'last',cast_to_int=True)
        cp.end_chain()
    return cp.compile()

# Summary statistics - Number of non-sparse elements
def summaryStats(cp):
    # Remember to properly initialize fu.vrf

    # Sum of all values
    cp.begin_chain()
    cp.vv_add(0,'notfirst')
    cp.v_cache(0)
    cp.v_reduce()
    cp.v_commit(1,'last')
    cp.end_chain()

    # Number of sparse elements
    cp.begin_chain()
    cp.vv_filter(0)
    cp.m_reduce('N')
    cp.vv_add(1,'notfirst')
    cp.v_cache(1)
    cp.v_reduce()
    cp.v_commit(1,'last')
    cp.end_chain()
    return cp.compile()

# Calculate spatial sparsity
def spatialSparsity(cp,N):
    # Remember to properly initialize fu.vrf
    cp.begin_chain()
    cp.vv_filter(0)
    cp.m_reduce('N')
    cp.v_commit(N,cast_to_int=True)
    cp.end_chain()
    return cp.compile()

# Check if previous vector changed
def vectorChange(cp):

    # Commit difference between current and previous sample
    cp.begin_chain()
    cp.vv_sub(0)
    cp.v_reduce()
    cp.v_commit(1)
    cp.end_chain()

    # Save vector to mem0
    cp.begin_chain()
    cp.v_cache(0)
    cp.end_chain()
    return cp.compile()

# Self correlation with the previous sample
def correlation(cp):

    # sum(X*Y) [Assuming that Y is stored in addr0]
    cp.begin_chain()
    cp.vv_mul(0)
    cp.v_reduce()
    cp.v_commit(1)
    cp.end_chain()

    # sum(X) [Storing X in addr0, which will become the Y of next vector]
    cp.begin_chain()
    cp.v_cache(0)
    cp.v_reduce()
    cp.v_commit(1)
    cp.end_chain()

    # sum(X*X)
    cp.begin_chain()
    cp.vv_mul(0)
    cp.v_reduce()
    cp.v_commit(1)
    cp.end_chain()
    return cp.compile()

# Check if previous vector changed
def passThrough(cp):
    cp.begin_chain()
    cp.end_chain()
    return cp.compile()

# Sum all input values
def sumAll(cp):
    cp.begin_chain()
    cp.v_reduce()
    cp.v_commit(1)
    cp.end_chain()
    return cp.compile()

# Raw values
def raw(cp):
    cp.begin_chain()
    cp.v_commit()
    cp.end_chain()
    return cp.compile()

# Simple test for vvalu
def vvalu_simple(cp):
    cp.begin_chain()
    cp.vv_add(0)
    cp.v_cache(0)
    cp.v_commit()
    cp.end_chain()
    return cp.compile()

# Simple test for fru
def fru_simple(cp):
    cp.begin_chain()
    cp.vv_filter(0)
    cp.m_reduce('M')
    cp.v_commit()
    cp.end_chain()
    return cp.compile()

# Multiple Chains
def multipleChains(cp):
    cp.begin_chain()
    cp.vv_filter(0)
    cp.m_reduce('M')
    cp.v_commit()
    cp.end_chain()

    cp.begin_chain()
    cp.v_reduce()
    cp.v_commit()
    cp.end_chain()

    cp.begin_chain()
    cp.v_commit()
    cp.end_chain()
    return cp.compile()

# Series of conditions for testing compiler
def conditions(cp):
    cp.begin_chain()
    cp.v_commit()
    cp.end_chain()

    cp.begin_chain()
    cp.vv_add(0,'notfirst')
    cp.v_commit()
    cp.end_chain()

    cp.begin_chain()
    cp.vv_add(0,'last')
    cp.v_cache(0)
    cp.v_commit(8,'notfirst')
    cp.end_chain()
    return cp.compile()

# Mini cache test
def minicache(cp):

    cp.begin_chain()
    cp.vv_add(0)
    cp.v_cache(0)
    cp.v_mc_save()
    cp.end_chain()

    cp.begin_chain()
    cp.v_mc_load()
    cp.vv_add(0)
    cp.v_cache(0)
    cp.v_mc_save()
    cp.end_chain()

    cp.begin_chain()
    cp.v_mc_load()
    cp.v_commit()
    cp.end_chain()
    return cp.compile()

# Activation Predictiveness
def activationPredictiveness(cp, elements_to_commit=1):
    # First we sum all activations of all nodes in address 0 (we will expect eof[0] to start a new sum)
    # Once we receive eof[0] we will check the max between this value and the one stored in the cache at address 1.
    # mc_save and mc_load are used to pass the average values from chain1 to chain2
    # Once we receive eof[1] we will commit a single value that corresponds to the max average of the values received.
    # The moving average is computed offline, since computing it on-chip will not reduce the amount of information that needs to be sent off-chip
    # For a better understanding of how this data gathering technique may be used, please refer to our software experiments

    cp.begin_chain()
    cp.v_reduce() 
    cp.vv_add(0,condition1='notfirst')
    cp.v_cache(0)
    cp.v_mc_save()
    cp.end_chain()

    cp.begin_chain()
    cp.v_mc_load()
    cp.vv_max(1,condition2='notfirst')
    cp.v_cache(1,condition1='last')
    cp.v_commit(elements_to_commit,condition2='last')
    cp.end_chain()

    return cp.compile()

# Norm Check
# To get better results, use FRU_reconfig_vector to change FRU's filter values according to the range of the percentiles (currently done via UART as a proof-of-concept)
# A single 64 bin distribution is used to get a proxy of the three percentiles, calculated offline.
def normCheck(cp,M):
    bins=64
    assert bins%M==0, "Number of bins must be divisible by M"
    for i in range(int(bins/M)):
        cp.begin_chain()
        cp.vv_filter(i)
        cp.m_reduce('M')
        cp.vv_add(i,'notfirst')
        cp.v_cache(i)
        cp.v_commit(M,'last')
        cp.end_chain()
    return cp.compile()
