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

from relax_spectrum.parameter import ScaleFactor
from relax_spectrum.mode import Constant

class Model:
    """A model of a spectrum.

    A model is a series of mode (`relax_spectrum.mode.Mode')


    Attributes
    ---------
    background: `relax_spectrum.mode.Mode'
        The background mode
    """

    def __init__(self):
        self._modes = [Constant("Default background", 0)]

    @property
    def background(self):
        return self._modes[0]

    def set_background(self, bg_mode):
        """Set the background mode."""
        self._modes[0] = bg_mode

    def add_peak(self, mode):
        """Add a peak to the model."""
        self._modes.append(mode)
        return mode

    def add_peaks(self, modes, scale=None):
        """Add a set of peaks to the model.

        Parameters
        ----------
        modes: list of `relax_spectrum.mode.Mode'
            The peaks to add the the model
        scale: float, optional
            Set the scale of each mode to this value.
        """
        for mode in modes:
            self._modes.append(mode)
            if scale is not None:
                if hasattr(mode, "scale"):
                    mode.scale.value = scale

    def __iter__(self):
        return self._modes.__iter__()

    def fix_background(self):
        """Fix the backround parameters"""
        for param in self._modes[0]:
            param.fixed = True

    def unfix_background(self):
        """Fix the backround parameters"""
        for param in self._modes[0]:
            param.fixed = True

    def fix_all(self):
        """Set all the parameters to be not fixed."""
        for mode in self._modes:
            for param in mode:
                param.fixed = False

    def unfix_all(self):
        """Set all the parameters to be not fixed."""
        for mode in self._modes:
            for param in mode:
                param.fixed = False


    def unfix_scale(self):
        """Set all the parameters to be not fixed."""
        for mode in self._modes:
            for param in mode:
                if isinstance(param, ScaleFactor):
                    param.fixed = False


    def eval(self, x):
        """Evaluate the model."""
        y = np.zeros_like(x)
        for mode in self._modes:
            y += mode.eval(x)
        return y

    def __str__(self):
        pstr = ""
        for mode in self._modes:
            pstr += "- {0}:\n".format(mode.name)
            for param in mode:
             pstr += "\t+ {0}: {1}\n".format(param.name, param.value)
        return pstr
