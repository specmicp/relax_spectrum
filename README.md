relax_spectrum
==============


A simple package to fit spectrometry experiments (e.g. Raman, Infrared, NMR in frequency domain ...).


Minimal example
---------------

~~~python
data = simple_csv_reader("mydata.dat")
BackgroundInteractiveBaselineFitter(data, model).fit()


peak_1 = mode.Lorentzian("1", 1, 238, 100)
peak_1.position.set_bounds(10) # +/- 10%
peak_2 = mode.Lorentzian("2", 1, 441, 50)
peak_1.position.set_bounds(10)

model.add_peaks((peak_1, peak_2))
model.unfix_scale()
fitter = LeastSquareFitter(data, model)
fitter.fit()

model.unfix_all()
fitter.fit()

print(model)
~~~

About
-----

Developed by Fabien Georget, at the [Laboratory of Construction Materials](https://www.epfl.ch/labs/lmc/), EPFL, Lausanne, Switzerland
