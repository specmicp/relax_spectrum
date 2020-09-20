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

import numpy as np

__all__ = ["Parameter", "NonNegativeParameter", "ScaleFactor", "Position", "PeakWidth"]

class Parameter:
    """A generic parameter for a peak in a spectrum.

    The parameter is fixed by default.

    Parameters
    ----------
    name: str
        The name of the parameter
    value: numerical type
        Default/Initial/Fixed value for the parameters
    lower_bound: same as value, optional
        Lower bound for the parameter
    upper_bound: same as value, optional
        Upper bound for the parameter


    Attributes
    ----------
    name: str
        Name of the parameter
    value:
        Current value of the parameter
    lower_bound:
        The lower bound of the parameter
    upper_bound:
        The upper bound of the parameter
    fixed: bool
        If True, the parameter is fixed and it will not be fitted
    """

    def __init__(self, name, value, lower_bound=None, upper_bound=None):
        self._name = name
        self._value = float(value)
        self._old_value = None
        self._fixed = True
        self._lb = self._fix_lb_for_scipy(lower_bound)
        self._ub = self._fix_ub_for_scipy(upper_bound)

    @property
    def name(self):
        return self._name

    def _fix_lb_for_scipy(self, lb):
        if lb is None:
            return -np.inf
        else:
            return lb

    def _fix_ub_for_scipy(self, ub):
        if ub is None:
            return np.inf
        else:
            return ub

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._old_value = self._value
        self._value = value

    @property
    def lower_bound(self):
        return self._lb

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        self._lb = self._fix_lb_for_scipy(lower_bound)

    @property
    def upper_bound(self):
        return self._ub

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        self._ub = self._fix_ub_for_scipy(upper_bound)

    def set_bounds(self, x):
        """Set bounds at +/- x% of value."""
        self.lower_bound = self.value*(1-x/100)
        self.upper_bound = self.value*(1+x/100)


    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        self._fixed = value

class NonNegativeParameter(Parameter):
    """A non negative parameter."""

    def __init__(self, name, value, lower_bound=None, upper_bound=None):
        if lower_bound is None:
            lower_bound = 0
        if upper_bound is None:
            upper_bound = np.inf
        super().__init__(name, value, lower_bound, upper_bound)

class ScaleFactor(NonNegativeParameter):
    """Parameter describing the scale of a peak."""

    def __init__(self, value, lower_bound=None, upper_bound=None):
        super().__init__("Scale", value, lower_bound, upper_bound)

class Position(NonNegativeParameter):
    """Parameter describing the location of a peak."""

    def __init__(self, value, lower_bound=None, upper_bound=None):
        super().__init__("Position", value, lower_bound, upper_bound)

class PeakWidth(NonNegativeParameter):
    """Parameter describing the width of a peak."""

    def __init__(self, value, lower_bound=None, upper_bound=None):
        super().__init__("Width", value, lower_bound, upper_bound)
