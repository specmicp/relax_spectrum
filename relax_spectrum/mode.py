# Copyright (c) 2020 Fabien Georget <fabien.georget@epfl.ch>, EPFL
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from abc import ABC, abstractmethod

import numpy as np
import numpy.polynomial as npol

from relax_spectrum.parameter import Parameter, ScaleFactor, Position, PeakWidth

__all__ = ["Mode",
           "Constant",
           "Chebyshev", "Polynomial",
           "Lorentzian", "Gaussian"
           ]

class Mode(ABC):
    """The base class for a mode.

    Parameters
    ----------
    name: str
        The name of the mode

    Attributes
    ----------
    name: str
        The name of the mode
    nb_parameters: int
        The number of parameter for this mode
    parameters: list of `relax_spectrum.parameter.Parameter'
        The list of parameters for this mode
    parameter_values: list of numerical values
        The list of parameter values

    """

    def __init__(self, name):
        self._name = name
        self._parameters = []

    @property
    def name(self):
        return self._name

    @property
    def nb_parameters(self):
        return len(self._parameters)

    @property
    def parameters(self):
        return self._parameters

    def _add_parameter(self, param):
        self._parameters.append(param)

    @property
    def parameter_values(self):
        return [parameter.value for parameter in self.parameters]

    def update(self, new):
        """Update the parameters from the values in new.

        Parameters
        ----------
        new: array_like, size nb_parameters
            New values for all the parameters
        """
        for parameter, value in zip(self.parameters,new):
            parameter.value = value

    def eval(self, x):
        """Calculate the peak using the stored parameters.

        Parameters
        ----------
        x: array_like
            compute the mode at these points
        """
        test_values = self.parameter_values
        return self.compute(x, *test_values)

    @abstractmethod
    def compute(self, x, *test_values):
        """Calculate the peak using the given parameters.

        Parameters
        ----------
        x: array_like
            Compute the mode at these points
        test_values: list of separated parameters
            Values for the parameters
        """
        pass

    def __iter__(self):
        return self._parameters.__iter__()


class Constant(Mode):
    """A constant mode.

    Can be used to implement known offset.
    """
    def __init__(self, name, constant):
        super().__init__(name)
        self.constant = constant

    def compute(self, x, *test_values):
        return self.constant

class Baseline(Mode):
    """A constant baseline.

    Need to be obtained before fitting the full model.
    """
    def __init__(self, name, values):
        super().__init__(name)
        self.values = values

    def compute(self, x, *test_values):
        return self.values

class Chebyshev(Mode):
    """Chebyshev polynomial.

    Typically used for background.

    Parameters
    ----------
    name: str
        The name of the mode
    coeffs: array_like, size n
        The coefficients of the polynomial of order n+1
    """
    def __init__(self, name, coeffs):
        super().__init__(name)
        for ind, coeff in enumerate(coeffs):
            self._add_parameter(Parameter("Coeff {0}".format(ind), coeff))

    def compute(self, x, *coeffs):
        return npol.chebyshev.chebval(x, coeffs)


class Polynomial(Mode):
    """A polynomial.

    Typically used for background.

    Parameters
    ----------
    name: str
        The name of the mode
    coeffs: array_like, size n
        The coefficients of the polynomial of order n+1
    """
    def __init__(self, name, coeffs):
        super().__init__(name)
        for ind, coeff in enumerate(coeffs):
            self._add_parameter(Parameter("Coeff {0}".format(ind), coeff))

    def compute(self, x, *coeffs):
        return npol.polynomial.polyval(x, coeffs)


class Lorentzian(Mode):
    """A Lorentzian.

    Parameters
    ----------
    name: str
        The name of the mode
    scale: float
        The scale factor for the peak
    position: float
        The position of the peak
    fwhm: float
        The full width at half maximum

    Attributes
    ----------
    scale: float
        The scale factor for the peak
    position: float
        The position of the peak
    fwhm: float
        The full width at half maximum
    """
    def __init__(self, name, scale=1, position=0, fwhm=1):
        super().__init__(name)

        self._add_parameter(ScaleFactor(scale))
        self._add_parameter(Position(position))
        self._add_parameter(PeakWidth(fwhm))

    @property
    def scale(self):
        return self._parameters[0]

    @property
    def position(self):
        return self._parameters[1]

    @property
    def fwhm(self):
        return self._parameters[2]

    def compute(self, x, A, p0, w):
        return A/(1.0+np.power((x-p0)/(w/2), 2))


class Gaussian(Mode):
    """A Gaussian.

    Parameters
    ----------
    name: str
        The name of the mode
    scale: float
        The scale factor for the peak
    position: float
        The position of the peak
    fwhm: float
        The full width at half maximum

    Attributes
    ----------
    scale: float
        The scale factor for the peak
    position: float
        The position of the peak
    fwhm: float
        The full width at half maximum
    """
    def __init__(self, name, scale=1, position=0, fwhm=1):
        super().__init__(name)

        self._add_parameter(ScaleFactor(scale))
        self._add_parameter(Position(position))
        self._add_parameter(PeakWidth(fwhm))

    @property
    def scale(self):
        return self._parameters[0]

    @property
    def position(self):
        return self._parameters[1]

    @property
    def fwhm(self):
        return self._parameters[2]

    def compute(self, x, A, p0, w):
        return A*np.exp(-np.log(2)*np.power((x-p0)/(w/2), 2))


class PseudoVoigt(Mode):
    """A Gaussian.

    Parameters
    ----------
    name: str
        The name of the mode
    scale: float
        The scale factor for the peak
    position: float
        The position of the peak
    fwhm: float
        The full width at half maximum

    Attributes
    ----------
    scale: float
        The scale factor for the peak
    position: float
        The position of the peak
    fwhm: float
        The full width at half maximum
    """
    def __init__(self, name, scale=1, position=0, fwhm=1, mix=1):
        super().__init__(name)

        self._add_parameter(ScaleFactor(scale))
        self._add_parameter(Position(position))
        self._add_parameter(PeakWidth(fwhm))
        self._add_parameter(Parameter("Mix", mix, 0.0, 1.0))

    @property
    def scale(self):
        return self._parameters[0]

    @property
    def position(self):
        return self._parameters[1]

    @property
    def fwhm(self):
        return self._parameters[2]

    @property
    def mix(self):
        return self._parameters[3]

    def compute(self, x, A, p0, w, eta):
        return A*(eta/(1.0+np.power((x-p0)/(w/2), 2)) + (1-eta)*np.exp(-np.log(2)*np.power((x-p0)/(w/2), 2)))
