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
This module contains methods related to the calculation of transition dipole
elements in atomic systems.  
"""

from numpy import sqrt
from scipy.constants import physical_constants
from qutip.angularmomentum import check_jm, check_angular_addition, manifold
from qutip.angularmomentum import clebsch_gordan as cg


__all__ = ["electric", "magnetic"]


def delta(x, y):
    """Simple delta function"""
    if x == y:
        return 1.
    else:
        return 0.
    

def check_angular_quantum_numbers(numbers):
    """Check that the angular momentum quantum numbers are valid"""
    [F, mF, J, I, L, S] = numbers
    check_jm(F, mF)
    check_angular_addition(F, J, I)
    check_angular_addition(J, L, S)
    return None


def extract_angular_quantum_numbers(numbers):
    """Method to ensure angular momentum quantum numbers are floats"""
    for index in range(len(numbers)):
        numbers[index] = float(numbers[index])
    return numbers
    

def electric(final, initial, polarization, dipole_moment = 1.):
    """
    """
    # Extract angular momentum quantum numbers from final, initial lists
    check_angular_quantum_numbers(final)
    check_angular_quantum_numbers(initial)
    [Ff, mFf, Jf, If, Lf, Sf] = extract_angular_quantum_numbers(final)
    [Fi, mFi, Ji, Ii, Li, Si] = extract_angular_quantum_numbers(initial)
    
    # Enforce angular momentum selection rules.  Electric dipole transitions 
    # change L and therefore cannot change I or S.
    if not ([If, Sf] == [Ii, Si]):
        return 0.
    
    # Enforce the parity selection rule.  Electric transitions change parity.
    if (Lf - Li) % 2 == 0:
        return 0.
    
    # Enforce polarization selection rule.
    if not (mFf - mFi) == polarization:
        return 0.
    
    # Compute matrix element as a sum of components in the decoupled basis.
    # The electric dipole moment of the atom is a sum over dipole moments of
    # the spin, orbital and nuclear contributions.  
    element = 0.
    for mIf in manifold(If):
        mJf = mFf - mIf
        if abs(mJf) > Jf:
            continue
        
        for mIi in manifold(Ii):
            mJi = mFi - mIi
            if abs(mJi) > Ji:
                continue
            
            # Angular term from F = J + I decomposition
            term_FJI = cg(Jf, If, mJf, mIf, Ff, mFf) * \
                cg(Ji, Ii, mJi, mIi, Fi, mFi)
            
            for mSf in manifold(Sf):
                mLf = mFf - mIf - mSf
                if abs(mLf) > Lf:
                    continue
                
                for mSi in manifold(Si):
                    mLi = mFi - mIi - mSi
                    if abs(mLi) > Li:
                        continue
                    
                    # Angular term from J = L + S decomposition
                    term_JLS = cg(Lf, Sf, mLf, mSf, Jf, mJf) * \
                        cg(Li, Si, mLi, mSi, Ji, mJi)
                    angular_term = term_JLS * term_FJI

                    element += angular_term
    
    return dipole_moment * element


def moment(j1, j2, m1, m2, polarization):
    """Routine used to calculate the magnitude of the matrix element 
    ``<j1 m1| J |j2 m2>``, where ``J = J_{+} + J_0 + J_{-}`` is an angular 
    momentum vector in the spherical basis.  Used to calculate magnetic
    moments in the decoupled basis.
    """
    try:
        check_jm(j1, m1)
        check_jm(j2, m2)
    except TypeError:
        return 0.
    
    # Check that the angular momentum quantum number is unchanged.  If it is 
    # changed, return zero.
    if not (j1 == j2):
        return 0.
    
    # Check if the change in magnetic quantum numbers is allowed by the field
    # polarization.  If not, return zero.
    q = float(m1 - m2)
    if not q == polarization:
        return 0.
    
    else:
        # Handle each of the polarization cases
        if polarization == 1.:
            return sqrt(j2 * (j2 + 1.) - m2 * (m2 + 1.))
        elif polarization == 0.:
            return float(m2)
        elif polarization == -1.:
            return sqrt(j2 * (j2 + 1.) - m2 * (m2 - 1.))
        else:
            return 0.
        

def magnetic(final, initial, polarization, nuclear_g_factor = 0.):
    """
    """
    check_angular_quantum_numbers(final)
    check_angular_quantum_numbers(initial)
    [Ff, mFf, Jf, If, Lf, Sf] = extract_angular_quantum_numbers(final)
    [Fi, mFi, Ji, Ii, Li, Si] = extract_angular_quantum_numbers(initial)
    
    # Enforce angular momentum conservation.  Magnetic transitions cannot 
    # change I, L, or S, but can change their sums, J = L + S and F = J + I
    if not ([If, Lf, Sf] == [Ii, Li, Si]):
        return 0.
    
    # Enforce the parity selection rule.  Magnetic transitions preserve parity.
    if (Lf - Li) % 2 == 1:
        return 0.
    
    if not (mFf - mFi) == polarization:
        return 0.
    
    # g-factors and magnetons in units of MHz/Gauss
    gs = physical_constants["electron g factor"][0]
    gl = -1.
    gi = nuclear_g_factor
    bohr_magneton = physical_constants["Bohr magneton in Hz/T"][0] * 1E-10
    nuclear_magneton = physical_constants["nuclear magneton in MHz/T"][0] * 1E-4
    
    # Compute matrix element as a sum of components in the decoupled basis.
    # The magnetic moment of the atom is a sum over magnetic moments of the
    # electron, orbital and nuclear contributions.  
    element = 0.
    for mIf in manifold(If):
        mJf = mFf - mIf
        if abs(mJf) > Jf:
            continue
        
        for mIi in manifold(Ii):
            mJi = mFi - mIi
            if abs(mJi) > Ji:
                continue
            
            # Compute the nuclear magnetic moment
            nuclear_term = - gi * nuclear_magneton * \
                moment(If, Ii, mIf, mIi, polarization)
            
            # Angular term from F = J + I decomposition
            term_FJI = cg(Jf, If, mJf, mIf, Ff, mFf) * \
                cg(Ji, Ii, mJi, mIi, Fi, mFi)
            
            for mSf in manifold(Sf):
                mLf = mFf - mIf - mSf
                if abs(mLf) > Lf:
                    continue
                
                for mSi in manifold(Si):
                    mLi = mFi - mIi - mSi
                    if abs(mLi) > Li:
                        continue
                    
                    # Compute the electron magnetic moment
                    spin_term = - gs * bohr_magneton * \
                        moment(Sf, Si, mSf, mSi, polarization)
                    
                    # Compute the orbital magnetic moment
                    orbital_term = - gl * bohr_magneton * \
                        moment(Lf, Li, mLf, mLi, polarization)
                        
                    # Enforce magnetic quantum number selection rules.  A 
                    # magnetic dipole can only change one of the decoupled
                    # magnetic quantum numbers at a time.
                    nuclear_moment = nuclear_term * delta(mSf, mSi) * delta(mLf, mLi)
                    spin_moment = spin_term * delta(mIf, mIi) * delta(mLf, mLi)
                    orbital_moment = orbital_term * delta(mIf, mIi) * delta(mSf, mSi)
                    
                    # Angular term from J = L + S decomposition
                    term_JLS = cg(Lf, Sf, mLf, mSf, Jf, mJf) * \
                        cg(Li, Si, mLi, mSi, Ji, mJi)
                    angular_term = term_JLS * term_FJI
                        
                    # Magnetic moment
                    magnetic_moment = angular_term * \
                        (spin_moment + orbital_moment + nuclear_moment )
                    element += magnetic_moment
    
    return element
