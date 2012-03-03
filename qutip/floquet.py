#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from scipy import *
from qutip.Qobj import *

from qutip.superoperator import *
from qutip.odesolve import *
from qutip.essolve import *
from qutip.steady import steadystate
from qutip.states import basis
from qutip.states import projection
from qutip.Odeoptions import Odeoptions
from qutip.propagator import propagator

def get_min_and_index(lst): 
    """
    Private function for obtaining min and max indicies.
    """
    minval,minidx = lst[0],0 
    for i,v in enumerate(lst[1:]): 
        if v < minval: 
            minval,minidx = v,i+1 
    return minval,minidx 

def propagator_steadystate(U):
    """
    Find the steady state for successive applications of the propagator :math:`U`.
    
    Arguments:
    
        `U` (:class:`qutip.Qobj`) Operator representing the propagator.
    
    Returns a :class:`qutip.Qobj` instance representing the steady-state vector.
    """

    evals,evecs = la.eig(U.full())

    ev_min, ev_idx = get_min_and_index(abs(evals-1.0))

    N = int(sqrt(len(evals)))

    evecs = evecs.T
    rho = Qobj(vec2mat(evecs[ev_idx]))

    return rho * (1 / real(rho.tr()))


def floquet_modes(H, T, H_args=None):
    """
    Calculate the initial Floquet modes Phi_alpha(0) for a driven system with
    period T.
    
    Returns a list of :class:`qutip.Qobj` instances representing the Floquet
    modes and a list of corresponding quasienergies, sorted by increasing
    quasienergy in the interval [-pi/T, pi/T]
         
    .. note:: Experimental
    """

    # get the unitary propagator
    U = propagator(H, T, [], H_args)

    # find the eigenstates for the propagator
    evals,evecs = la.eig(U.full())

    eargs = angle(evals)
    
    # make sure that the phase is in the interval [-pi, pi], so that the
    # quasi energy is in the interval [-pi/T, pi/T] where T is the period of the
    # driving.
    #eargs  += (eargs <= -2*pi) * (2*pi) + (eargs > 0) * (-2*pi)
    eargs  += (eargs <= -pi) * (2*pi) + (eargs > pi) * (-2*pi)
    e_quasi = -eargs/T

    # sort by the quasi energy
    order = argsort(-e_quasi)

    # prepare a list of kets for the floquet states
    new_dims  = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(matrix(evecs[:,o]).T, dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]

def floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args=None):
    """
    Calculate the Floquet modes at times tlist Phi_alpha(tlist) propagting the
    initial Floquet modes Phi_alpha(0)
    
    .. note:: Experimental    
    """

    # find t in [0,T] such that t_orig = t + n * T for integer n
    t = t - int(t/T) * T
    
    f_modes_t = []
        
    # get the unitary propagator from 0 to t
    if t > 0.0:
        U = propagator(H, t, [], H_args)

        for n in arange(len(f_modes_0)):
            f_modes_t.append(U * f_modes_0[n] * exp(1j * f_energies[n]*t))

    else:

        f_modes_t = f_modes_0
        #for n in arange(len(f_states_0)):
        #    f_modes_t.append(f_states_0[n])


    return f_modes_t
    
def floquet_states_t(f_modes_0, f_energies, t, H, T, H_args=None):
    """
    Evaluate the floquet states at time t.
    
    Returns a list of the wavefunctions.
        
    .. note:: Experimental    
    """
    
    f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args)
    return [(f_modes_t[i] * exp(-1j * f_energies[i]*t)) for i in arange(len(f_energies))]    
    
    
def floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, H_args=None):
    """
    Evaluate the wavefunction for a time t using the Floquet states decompositon.
    
    Returns the wavefunction.
        
    .. note:: Experimental    
    """
    
    f_states_t = floquet_states_t(f_modes_0, f_energies, t, H, T, H_args)
    return sum([f_states_t[i] * f_coeff[i] for i in arange(len(f_energies))])

def floquet_state_decomposition(f_modes_0, f_energies, psi0):
    """
    Decompose the wavefunction psi in the Floquet states, return the coefficients
    in the decomposition as an array of complex amplitudes.
    """
    return [(f_modes_0[i].dag() * psi0).data[0,0] for i in arange(len(f_energies))]

# should be moved to a utility library?    
def n_thermal(w, w_th):
    if (w > 0): 
        return 1.0/(exp(w/w_th) - 1.0)
    else: 
        return 0.0
    
def floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T, H_args, J_cb, w_th, kmax=5):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.
    """
    
    N = len(f_energies)
    M = 2*kmax + 1
    
    omega = (2*pi)/T
    
    Delta = zeros((N, N, M))
    X     = zeros((N, N, M), dtype=complex)
    Gamma = zeros((N, N, M))
    A     = zeros((N, N))
    
    nT = 100
    dT = T/nT
    tlist = arange(dT, T+dT/2, dT)

    for t in tlist:
        # TODO: repeated invocations of floquet_modes_t is inefficient...
        # make a and b outer loops and use the odesolve instead of the propagator.
        f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args)   
        for a in range(N):
            for b in range(N):
                k_idx = 0
                for k in range(-kmax,kmax+1, 1):
                    X[a,b,k_idx] += (dT/T) * exp(-1j * k * omega * t) * (f_modes_t[a].dag() * c_op * f_modes_t[b]).full()[0,0]
                    k_idx += 1

    Heaviside = lambda x: ((sign(x)+1)/2.0)
    for a in range(N):
        for b in range(N):
            k_idx = 0
            for k in range(-kmax,kmax+1, 1):
                Delta[a,b,k_idx] = f_energies[a] - f_energies[b] + k * omega
                Gamma[a,b,k_idx] = 2 * pi * Heaviside(Delta[a,b,k_idx]) * J_cb(Delta[a,b,k_idx]) * abs(X[a,b,k_idx])**2
                k_idx += 1
                

    for a in range(N):
        for b in range(N):
            for k in range(-kmax,kmax+1, 1):
                k1_idx =   k + kmax;
                k2_idx = - k + kmax;                
                A[a,b] += Gamma[a,b,k1_idx] + n_thermal(abs(Delta[a,b,k1_idx]), w_th) * (Gamma[a,b,k1_idx]+Gamma[b,a,k2_idx])
                
    return Delta, X, Gamma, A
        
    
def floquet_collapse_operators(A):
    """
    Construct
    """
    c_ops = []
    
    N, M = shape(A)
    
    #
    # Here we really need a master equation on Bloch-Redfield form, or perhaps
    # we can use the Lindblad form master equation with some rotating frame
    # approximations? ...
    # 
    for a in range(N):
        for b in range(N):
            if a != b and abs(A[a,b]) > 0.0:
                # only relaxation terms included...
                c_ops.append(sqrt(A[a,b]) * projection(N, a, b))
    
    return c_ops
    
    
def floquet_master_equation_tensor(Alist):
    """
    Construct a tensor that represents the master equation in the floquet
    basis (with constant Hamiltonian and collapse operators).
    
    Simplest RWA appriximation [Grifoni et al, Phys.Rep. 304 229 (1998)]
    """

    if isinstance(Alist, list):
        # Alist can be a list of rate matrices corresponding
        # to different operators that couple to the environment
        N, M = shape(Alist[0])
    else:
        # or a simple rate matrix, in which case we put it in a list
        Alist = [Alist]
        N, M = shape(Alist[0])
      
    R = Qobj(scipy.sparse.csr_matrix((N*N,N*N)), [[N,N], [N,N]], [N*N,N*N])

    R.data=R.data.tolil()      
    for I in range(N*N):
        a,b = vec2mat_index(N, I)
        for J in range(N*N):
            c,d = vec2mat_index(N, J)
        
            for A in Alist:               
                s1 = s2 = 0
                for n in range(N):
                    s1 += A[a,n] * (n == c) * (n == d) - A[n,a] * (a == c) * (a == d)
                    s2 += (A[n,a] + A[n,b]) * (a == c) * (b == d)
                       
                dR = (a == b) * s1 - 0.5 * (1 - (a == b)) * s2
                            
                if dR != 0.0:
                    R.data[I,J] += dR

    R.data=R.data.tocsr()
    return R                   

    
def floquet_master_equation_steadystate(H, A):
    """
    Returns the steadystate density matrix (in the floquet basis!) for the
    Floquet-Markov master equation.
    """
    c_ops = floquet_collapse_operators(A)
    rho_ss = steadystate(H, c_ops)   
    return rho_ss
    
    
def floquet_basis_transform(f_modes, f_energies, rho0):
    """
    Make a basis transform that takes rho0 from the floquet basis to the 
    computational basis.
    """
    return rho0.transform(f_modes, True)

#-------------------------------------------------------------------------------
# Floquet-Markov master equation
# 
# 
def fmmesolve(R, ekets, rho0, tlist, e_ops, opt=None):
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.
    
    ..note:: Experimental
    
    """

    if opt == None:
        opt = Odeoptions()
        opt.nsteps = 2500  # 

    if opt.tidy:
        R = tidyup(R, opt.atol)

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()
       
    #
    # prepare output array
    # 
    n_e_ops  = len(e_ops)
    n_tsteps = len(tlist)
    dt       = tlist[1]-tlist[0]

    if n_e_ops == 0:
        result_list = []
    else:
        result_list = []
        for op in e_ops:
            if op.isherm and rho0.isherm:
                result_list.append(zeros(n_tsteps))
            else:
                result_list.append(zeros(n_tsteps,dtype=complex))


    #
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis
    #
    if ekets != None:
        rho0 = rho0.transform(ekets)
        for n in arange(len(e_ops)):
            e_ops[n] = e_ops[n].transform(ekets)

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cyq_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, #nsteps=opt.nsteps,
                              #first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        rho.data = vec2mat(r.y)
        
        # calculate all the expectation values, or output rho if no operators
        if n_e_ops == 0:
            result_list.append(Qobj(rho))
        else:
            for m in range(0, n_e_ops):
                result_list[m][t_idx] = expect(e_ops[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list

    