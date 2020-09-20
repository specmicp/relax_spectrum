import numpy as np

from relax_spectrum.model import Model
from relax_spectrum.data import simple_csv_reader
from relax_spectrum.fitter import LeastSquareFitter, BackgroundPreFit, \
 BackgroundBaselineFitter, BackgroundInteractiveBaselineFitter

import relax_spectrum.mode as mode

import matplotlib.pyplot as plt

model = Model()


data = simple_csv_reader("test/rutile.dat", delimiter=",")
BackgroundInteractiveBaselineFitter(data, model).fit()


peak_s1 = mode.Lorentzian("s1", 1, 125, 10)
peak_s1.position.set_bounds(10)
peak_s2 = mode.Lorentzian("s2", 1, 145, 10)
peak_s2.position.set_bounds(10)
peak_1 = mode.Lorentzian("1", 1, 238, 100)
peak_1.position.set_bounds(10)
peak_2 = mode.Lorentzian("2", 1, 441, 50)
peak_2.position.set_bounds(10)
peak_3 = mode.Lorentzian("3", 1, 607, 50)
peak_3.position.set_bounds(10)


model.add_peaks((peak_s1,peak_s2,peak_1, peak_2, peak_3))

model.unfix_scale()


fitter = LeastSquareFitter(data, model)

fitter.fit()

model.unfix_all()
fitter.fit()


print(model)


fig, ax = plt.subplots()
ax.plot(data.x, data.y, "red")


ax.plot(data.x, model.eval(data.x), color="blue")

for amode in model:
    ax.plot(data.x, amode.eval(data.x), "--", color="black")

fig, ax = plt.subplots()
ax.plot(data.x, data.y-model.background.values, "red")


ax.plot(data.x, model.eval(data.x), color="blue")

for amode in model._modes[1:]:
    ax.plot(data.x, amode.eval(data.x), "--", color="black")


plt.show()