# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
This module computes propagators with a Trotter-Suzuki algorithm, also called
time-evolving block decimation
"""

import types
import numpy as np
import scipy.linalg as la

from qutip import expm
from qutip.qobj import Qobj
from qutip.superoperator import vec2mat, mat2vec


__all__ = ["trotter"]


def _list_format_to_function(H):
    """
    Private function to convert from list specification format to function
    format.
    """
    
    # Construct a function-based Hamiltonian
    def Hamiltonian(t, args):
        Hamiltonian_ = 0
        for item in H:
            
            # Check if item is a sublist
            if isinstance(item, list):
                if isinstance(item[1], types.FunctionType):
                    Hamiltonian_ += item[0] * item[1](t, args)
                else:
                    prompt = "The [qobj, func] pair %s does not have a " + \
                        "callable func method"
                    prompt = prompt %(str(item))
                    raise ValueError(prompt)
            elif isinstance(item, qobj):
                Hamiltonian_ += item
        return Hamiltonian_
        
    return Hamiltonian


def trotter(H, t, c_op_list, H_args=None):
    """
    Calculate the propagator U(t) for the density matrix or wave function such
    that :math:`\psi(t) = U(t)\psi(0)` or
    :math:`\\rho_{\mathrm vec}(t) = U(t) \\rho_{\mathrm vec}(0)`
    where :math:`\\rho_{\mathrm vec}` is the vector representation of the
    density matrix.
    """

    tlist = [0, t] if isinstance(t, (int, float, np.int64, np.float64)) else t

    if len(c_op_list) == 0:
        
        if isinstance(H, types.FunctionType):
            # H is a function based Hamiltonian
            H0 = H(0.0, H_args)
            N = H0.shape[0]
            dims = H0.dims
            Hfunc = H
        elif isinstance(H, list):
            # H is in list specification format
            if isinstance(H[0], list):
                H0 = H[0][0]
            else:
                H0 = H[0]
            N = H0.shape[0]
            dims = H0.dims
            Hfunc = _list_format_to_function(H)
        else:
            # H is a simple operator
            N = H.shape[0]
            dims = H.dims
            Hfunc = lambda t, args: H

        # Compute the propagator using the Trotter series.
        U = Qobj(np.identity(N))
        Dt = np.diff(tlist)
        Data = [Qobj(U, dims=dims)]
        
        for n in range(len(Dt)):
            dt = Dt[n]
            t = tlist[n]
            H = Hfunc(t, H_args)
            Ut = Qobj(expm( -1j * H.data * dt ))
            U = Ut * U
            Data.append(Qobj(U, dims=dims))

        return Data

    else:
        #
        # TODO: implement a Trotter series for a Liovillian master equation
        #
        return NotImplemented



