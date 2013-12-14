# ex_trotter_vs_mesolve.py
#
# Copyright (c) 2013 True Merrill
# Georgia Tech Research Institute
# 
# This is a test script to determine the relative speeds of integrators used to
# calculate quantum propagators.  The standard QuTiP method `propagator`
# calculates the quantum propagator by using `mesolve` for each basis state,
# propagating these forward in time, and constructing the total propagator row
# by row.  In contrast my new method `qutip.trotter` calculates the propagator
# using a Trotter series.

from qutip import *
from pylab import *
import timeit

def Wrapper(func, *args, **kwargs):
    def Wrapped():
        return func(*args, **kwargs)
    return Wrapped


def RunBenchmarking(H, t, c_op_list, H_args):
    """Run a benchmarking run, comparing Propagator to Trotter"""
    
    number = 5
    for func in [propagator, trotter]:
        print("Testing %s" %(str(func)))
        wrappedFunc = Wrapper(func, H, t, c_op_list, H_args) 
        result = timeit.timeit(
            wrappedFunc,
            number = number  
        )
        print("Average time %.4f seconds" %(result/number))
    
    return


def RandomHamiltonian(dimension):
    """Create a random Hermitian operator (eg a Hamiltonian)"""
    
    # Create two random matrices.
    A = matrix( random((dimension, dimension)) )
    B = matrix( random((dimension, dimension)) )
    
    # Compute a random hermetian matrix
    H = (A + A.H) + 1j * (B - B.H)
    
    return Qobj(H)


if __name__ == "__main__":
    
    # Create a random Hamiltonian
    dimension = 4
    H = RandomHamiltonian(4)
    times = linspace(0, 5, 100)
    
    def Coefficient(t, args):
        # Time-dependent coefficient for the Hamiltonian.  Since we choose a
        # coefficient equal to sin(t), we can compute the propagator
        # analytically as
        #
        # U(t) = exp( -i (1 - cos(t)) H )
        #
        return sin(t)
    
    def AnalyticalResult(t):
        return Qobj(expm(-1j * (1-cos(t)) * H.full()))
    
    RunBenchmarking(H, times, [], [])
    
    # Compare the accuracy of the Trotter method and the mesolve to the
    # analytical result
    print("Computing trace-distance with analytical result")
    
    def TraceDistance(U, V):
        return trace(U.full().H -V.full())
    
    propagatorMesolve = propagator(H, times, [])
    propagatorTrotter = trotter(H, times, [])
    distance = zeros((6, len(times)))
    
    for index in range(len(times)):
        t = times[index]
        U = AnalyticalResult(t)
        Ut = propagatorTrotter[index]
        Um = propagatorMesolve[index]
        
        distance[0, index] = abs(Ut[0,0])**2
        distance[1, index] = abs(Um[0,0])**2
        distance[2, index] = abs(Ut[1,1])**2
        distance[3, index] = abs(Um[1,1])**2
        distance[4, index] = abs(U[0,0])**2
        distance[5, index] = abs(U[1,1])**2
    
    # Plot the errors
    print("Plotting")
    plot(
        times, distance[0,:], "r-s",
        times, distance[1,:], "ro",
        times, distance[2,:], "b-s",
        times, distance[3,:], "bo"
    )
        #times, distance[4,:], "k",
        #times, distance[5,:], "k"
    #)
    show()