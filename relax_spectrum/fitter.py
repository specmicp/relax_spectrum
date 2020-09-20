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
from numpy.linalg import norm

import scipy.optimize as sco
from scipy import sparse
from scipy.sparse.linalg import spsolve

from relax_spectrum.mode import Baseline

import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button, RadioButtons

__all__ = ["Fitter", "BackgroundPreFit", "BackgroundBaselineFitter", "LeastSquareFitter"]

class Fitter(ABC):
    """Base class for a fitter.

    Attributes
    ----------
    model: `relax_spectrum.model.Model'
        The model to fit
    data: `relax_spectrum.data.Data'
        The data to fit

    Parameter
    ---------
    model: `relax_spectrum.model.Model'
        The model to fit
    data: `relax_spectrum.data.Data'
        The data to fit
    """

    def __init__(self, data, model):
        self._model = model
        self._data = data

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @abstractmethod
    def fit(self):
        pass

def baseline_als(y, lamb, p, niter=10):
    """Background fitting algorithm.

    Adapted from:
     "Baseline Correction with Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens, 2005
     https://stackoverflow.com/questions/29156532/python-baseline-correction-library/29185844
    """
    n = len(y)

    # the penalization term
    D = sparse.diags([1,-2,1],[0,1,2], shape=(n-2,n))
    D = lamb*(D.transpose()).dot(D)

    # weights vector
    w = np.ones(n)
    W = sparse.spdiags(w, 0, n, n)

    for i in range(niter):
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
        W.setdiag(w)
    return z

def baseline_airpls(signal, lamb):
    """
    From

    Zhang, Z.-M., Chen, S., Liang, Y.-Z., 2010.
    Baseline correction using adaptive iteratively reweighted penalized least squares. Analyst 135, 1138–1146.
    https://doi.org/10.1039/B922045C
    """
    n = len(signal)
    max_iter = 100

    # the penalization term
    D = sparse.diags([1,-2,1],[0,1,2], shape=(n-2,n))
    D = lamb*(D.transpose()).dot(D)

    # weights vector
    w = np.ones(n)
    W = sparse.spdiags(w, 0, n, n)

    nb_iter = 0
    retcode = -1
    while True:
        z = spsolve(W+D, w*signal)
        d = signal-z
        dn = np.where(d>=0,0,d)
        #
        if norm(dn) < 0.001*norm(signal):
            retcode = 1
            break
        if nb_iter >= max_iter:
            retcode = 0
            break
        w = np.where(d>=0,0,np.exp(nb_iter*dn/norm(dn)))
        W.setdiag(w)

    return z


class BackgroundBaselineFitter(Fitter):
    """A simple fitter setting the background to fix values."""
    def fit(self, lam, p):
        z = baseline_als(self.data.y, lam, p)
        print(z)
        self.model.set_background(Baseline("Fixed background", z))

class BackgroundInteractiveBaselineFitter(Fitter):
    """A simple fitter setting the background to fix values."""
    def fit(self):
        return self.fit_airpls()

    def fit_airpls(self):

        z = baseline_airpls(self.data.y, 1e6,)
        fig, (ax, axlamb, axaccept) = plt.subplots(3,1, gridspec_kw={"height_ratios": [10,1,1]})
        ax.plot(self.data.x, self.data.y)

        line,  = ax.plot(self.data.x, z)


        ax.margins(x=0)

        slamb = Slider(axlamb, 'log lambda', 1, 10.0, valinit=6, valstep=0.05)
        baccept = Button(axaccept, 'Accept')

        def update(val):
            loglamb = slamb.val
            z = baseline_airpls(self.data.y, 10**loglamb)
            line.set_ydata(z)
            fig.canvas.draw_idle()

        def close(val):
            plt.close(fig)

        slamb.on_changed(update)
        baccept.on_clicked(close)

        plt.show(block=True)

        loglamb = slamb.val
        z = baseline_airpls(self.data.y, 10**loglamb)
        self.model.set_background(Baseline("Fixed background", z))

    def fit_als(self):

        z = baseline_als(self.data.y, 1e6, 0.01)
        fig, (ax, axfreq, axamp) = plt.subplots(3,1, gridspec_kw={"height_ratios": [10,1,1]})
        ax.plot(self.data.x, self.data.y)

        line,  = ax.plot(self.data.x, z)


        ax.margins(x=0)

        #axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
        #axamp = plt.axes([0.25, 0.15, 0.65, 0.03])

        sfreq = Slider(axfreq, 'log lambda', 1, 10.0, valinit=6, valstep=0.05)
        samp = Slider(axamp, 'log p', -10.0, -1, valinit=-2, valstep=0.05)


        def update(val):
            amp = samp.val
            freq = sfreq.val
            z = baseline_als(self.data.y, 10**freq, 10**amp)
            line.set_ydata(z)
            fig.canvas.draw_idle()


        sfreq.on_changed(update)
        samp.on_changed(update)

        plt.show(block=True)

        amp = samp.val
        freq = sfreq.val
        z = baseline_als(self.data.y, 10**freq, 10**amp)
        self.model.set_background(Baseline("Fixed background", z))


class BackgroundPreFit(Fitter):
    """A simple fitter to get a reasonnable background.

    Use an asymmetric least squares smoothing."""
    def fit(self, lam, p):
        """Fit the background.

        Parameter
        --------
        lam: float
            Smoothness parameter (usually 10@2 to 10^9)
        p: float
            Asymmetry parameter (usually 0.01 to 0.001)
        """
        bg_mode = self.model.background
        z = baseline_als(self.data.y, lam, p)
        popt, _ = sco.curve_fit(bg_mode.compute, self.data.x, z, bg_mode.parameter_values)
        bg_mode.update(popt)
        return z

class LeastSquareFitter(Fitter):
    """A nonlinear least square fitter

    Based on `scipy.optimize.least_squares'

    Attributes
    ----------
    model: `relax_spectrum.model.Model'
        The model to fit
    data: `relax_spectrum.data.Data'
        The data to fit
    x_tolerance: float
        Tolerance on the variable, default 1e-8
    f_tolerance: float
        Tolerance on the residual, default 1e-8

    Parameter
    ---------
    model: `relax_spectrum.model.Model'
        The model to fit
    data: `relax_spectrum.data.Data'
        The data to fit

    """
    def __init__(self, data, model):
        super().__init__(data, model)
        self._xtol = 1e-8
        self._ftol = 1e-8

    @property
    def x_tolerance(self):
        return self._xtol

    @x_tolerance.setter
    def x_tolerance(self, value):
        self._ftol = self.value

    @property
    def f_tolerance(self):
        return self._xtol

    @f_tolerance.setter
    def f_tolerance(self, value):
        self._ftol = self.value

    def _number_parameters(self):
        """Number the parameters to include in the fit."""
        cur = 0
        param_list = []
        init = []
        lb = []
        ub = []
        for mode in self.model:
            l = []
            for param in mode:
                if param.fixed:
                    l.append(-1)
                else:
                    l.append(cur)
                    init.append(param.value)
                    lb.append(param.lower_bound)
                    ub.append(param.upper_bound)
                    cur += 1
            param_list.append(l)

        if cur == 0:
            raise RuntimeError("No parameter to fit !")

        self._param_number =  param_list

        return np.asarray(init), (np.asarray(lb), np.asarray(ub))

    def _get_objective_function(self):
        """Return the objective function to minimize."""
        def objective(params):
            value = -self.data.y
            # compute model
            for mode, nbmode in zip(self.model,self._param_number):
                test_params = []
                for param, ideq in zip(mode,nbmode):
                    if ideq == -1:
                        test_params.append(param.value)
                    else:
                        test_params.append(params[ideq])
                value += mode.compute(self.data.x, *test_params)
            return value
        return objective

    def _update_parameters(self, x):
        """Update the parameters of the model."""
        for mode, nbmode in zip(self.model,self._param_number):
                for param, ideq in zip(mode,nbmode):
                    if ideq > -1:
                        param.value = x[ideq]

    def fit(self):
        x0, bounds = self._number_parameters()
        # get residuals
        obj = self._get_objective_function()
        # call scipy.least_squares
        res = sco.least_squares(obj, x0, bounds=bounds)#, loss="soft_l1")
        # TODO check
        # params
        self._update_parameters(res.x)
        return res