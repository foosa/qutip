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
This module defines the Laser class, which stores information related to laser
fields.  
"""

import numpy
from scipy.constants import physical_constants

#
# Speed of light: We need the speed of light in vacuum in order to enforce the
# dispersion relationship, ``\omega = c |k|``.
#
SpeedOfLight = physical_constants["speed of light in vacuum"][0]

class Laser:
    """
    Class representation of a laser field used to couple quantum states.

    Attributes
    ----------
    Amplitude : float
        Electric field amplitude of the laser field.  In usual units the
        electric field is measured in V/m.
    Phase : float
        Electromagnetic phase of the laser.
    Frequency : float
        Frequency of the laser field.  In usual units the frequency is in MHz.
    Wavevector: ndarray
        Array of Cartesian components for the laser wavevector.  The wavevector
        is renormalized at initialization to enforce the dispersion relationship.
    Polarization: ndarray
        Array of spherical components of the laser polarization, referenced to
        the quantization axis.  The polarization vector is renormalized at
        initialization.

    Methods
    -------
    __call__(coordinates)
        Returns the value of the electric field at coordinates.  When the
        coordinate is a float or a length-one array, the coordinate is
        interpreted as a time coordinate.  Alternatively, when coordinate is a
        length-four array, each component is interpreted as coordinates of a
        four-vector.
    __normalize_wavevector()
        Renormalize the wavevector to satisfy the dispersion relation
    __normalize_polarization()
        Renormalize the polarization vector
    """
    
    def __init__(self, **kwargs):
        
        default_keys = {
            "Amplitude" : 1.0,
            "Phase" : 0.0,
            "Frequency" : 1.0,
            "Wavevector" : numpy.array([0, 0, 1]),
            "Polarization" : numpy.array([1, 0, 1])
        }

        # Check if user supplied keyword arguments for the laser parameters.  If
        # a required parameter is missing, then use the default parameter.
        for key in kwargs:
            if not hasattr(self, key):
                setattr(self, key, kwargs[key])

        for key in default_keys:
            if not hasattr(self, key):
                setattr(self, key, default_keys[key])

        # Renormalize the wavevector and polarization
        if not len(self.Wavevector) == 3:
            raise ValueError("The wavevector must have three components.")
        
        if not len(self.Polarization) == 3:
            raise ValueError(
                "The polarization vector must have three components.")

        self.__normalize_wavevector()
        self.__normalize_polarization()
        return None
 

    def __call__(self, coordinates):
        """Method to return the value of the electric field evaluated at the
        ``coordinates``.

        Parameters
        ----------
        coordinates : ndarray
            Coordinates to evaluate the electric field.  When ``coordinate`` is
            an array with four elements, the coordinates are treated as a
            four-vector, ``(t, x, y, z) = coordinates``.  When ``coordinate`` 
            is an array with a single element, the single component is treated
            as a time coordinate and the spacial coordinates are all set to
            zero.

        Returns
        -------
        value : float
            Value of the electric field sampled at ``coordinates``
        """
        if not hasattr(coordinates, "__len__"):
            coordinates = numpy.array([coordinates])
        else:
            coordinates = numpy.asarray(coordinates)

        if len(coordinates) == 4:
            # Treat coordinates as a four-vector
            four_vector = numpy.array(coordinates)
        elif len(coordinates) == 1:
            # Only a time value was given, set spacial coordinates to zero.
            four_vector = numpy.array(
                [coordinates[0], 0, 0, 0]
                )
        else:
            raise ValueError("Coordinates must have either 1 or 4 components.")

        four_momentum = numpy.concatenate(
            [-numpy.array([self.Frequency]), numpy.array(self.Wavevector)]
            )
        four_phase = numpy.dot(four_vector, four_momentum) + self.Phase
        
        return self.Amplitude * numpy.exp( 1j * four_phase )


    def __normalize_wavevector(self):
        """Method to renormalize ``self.Wavevector`` so that the dispersion
        relationship ``\omega = c |k|`` is enforced.
        """
        magnitude = self.Frequency / SpeedOfLight
        normal = self.Wavevector / numpy.linalg.norm(self.Wavevector)
        self.Wavevector = magnitude * normal
        return None
        

    def __normalize_polarization(self):
        """Method to renormalize ``self.Polarization`` so that the polarization
        vector is a unit vector.
        """
        normal = self.Polarization / numpy.linalg.norm(self.Polarization)
        self.Polarization = normal
        return None

