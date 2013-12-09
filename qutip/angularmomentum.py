# This file is part of J. True Merrill's additions to QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2013 and later, J. True Merrill
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
"""
This module contains methods related to the addition of angular momentum, 
Clebsch-Gordan coefficients, and the Wigner-Eckart theorem.
"""

import scipy
import numpy
from math import factorial, floor

__all__ = ["check_jm", "check_angular_addition", "manifold", "wigner3j", 
           "clebsch_gordan"]


def check_jm(j, m):
    """Method to check if j, m are good angular momentum quantum numbers"""
    if (scipy.fix(2 * j) != 2 * j) or (j < 0):
        raise TypeError('j = %.1f must be a non-negative integer or half-integer' %(j))
    if (scipy.fix(2 * m) != 2 * m):
        raise TypeError('m = %.1f must be an integer or half-integer' %(m))
    if not (j % 1 == m % 1):
        raise TypeError('j = %.1f, m = %.1f must both be either integers or half-integers' %(j, m))
    if (abs(m) > j):
        raise TypeError('|m| must not exceed j') 
    return


def check_angular_addition(j, j1, j2):
    """Method to check whether j = j1 + j2 can be satisfied using addition of
    angular momentum rules"""
    if (scipy.fix(2 * j) != 2 * j) or (j < 0):
        raise TypeError("j must be a non-negative integer or half-integer")
    if (scipy.fix(2 * j1) != 2 * j1) or (j1 < 0):
        raise TypeError("j1 must be a non-negative integer or half-integer")
    if (scipy.fix(2 * j2) != 2 * j2) or (j2 < 0):
        raise TypeError("j2 must be a non-negative integer or half-integer")
    if j < abs(j1 - j2) or j > j1 + j2:
        raise ValueError("j = j1 + j2 cannot be satisfied")
    return
    

def manifold(j):
    """Generates an array of mj values"""
    return numpy.arange(j, -(j+1), -1)


def wigner3j(j1, j2, j, m1, m2, m):
    """Calculates the Wigner 3j symbol used in addition of angular momentum.

    Parameters
    ----------
    j1, j2, j : float
        Angular momentum quantum number. Must be integer or half-integer.

    m1, m2, m : float
        Magnetic quantum number. Must be integer or half-integer.

    Returns
    -------
    wigner : float
      float representing the Wigner-3j symbol ``(j1 j2 j m1 m2 m)``.
    """
    # Check angular momentum quantum numbers
    check_jm(j1, m1)
    check_jm(j2, m2)
    check_jm(j, m)

    # Calculate coeffecient
    t1 = j2 - m1 - j
    t2 = j1 + m2 - j
    t3 = j1 + j2 - j
    t4 = j1 - m1
    t5 = j2 + m2
    tmin = int( max( 0, max( t1, t2)) )
    tmax = int( min( t3, min( t4, t5)) )
    
    wigner = 0.
    
    for t in range(tmin, tmax + 1):
        wigner = wigner + (-1)**t / float( factorial(t) * factorial(t-t1) * \
        factorial(t-t2) * factorial(t3-t) * factorial(t4-t) * factorial(t5-t))

    wigner = wigner * (-1)**(j1-j2-m) * numpy.sqrt( factorial(j1+j2-j) * \
    factorial(j1-j2+j) * factorial(-j1+j2+j) / float(factorial(j1+j2+j+1)) * \
    factorial(j1+m1) * factorial(j1-m1) * factorial(j2+m2) * \
    factorial(j2-m2) * factorial(j+m) * factorial(j-m) )
    
    return wigner
    

def clebsch_gordan(j1, j2, m1, m2, j, m):
    """Calculates the Clebsch-Gordan coefficient ``<j1, j2, m1, m2 | j, m>``.

    Parameters
    ----------
    j1, j2, j : float
        Angular momentum quantum number. Must be integer or half-integer.

    m1, m2, m : float
        Magnetic quantum number. Must be integer or half-integer.

    Returns
    -------
    cg : float
      float representing the Clebsch-Gordan coefficient ``<j1,j2,m1,m2|j,m>``.
    """
    if m1 + m2 != m:
        return 0.
    if (j1-m1) != floor(j1-m1) or (j2-m2) != floor(j2-m2) or \
        (j-m) != floor(j-m):
        return 0.
    if j > j1 + j2 or j < abs(j1-j2):
        return 0.
    if abs(m1) > j1 or abs(m2) > j2 or abs(m) > j:
        return 0.
        
    # Calculate the Clebsch-Gordan coefficient from the Wigner-3j symbol
    cg = (-1)**(j1-j2+m) * numpy.sqrt(2*j + 1) * wigner3j(j1, j2, j, m1, m2, -m)
    return cg
    
