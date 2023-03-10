# Angle Generation
This directory contains three jupyter notebooks that demonstrate the new angle generation techniques that will be used for calculating the angles necessary to implement QSP. This was predominantly done so that we can encode more complicated functions, allowing us to extend QSP from Hamiltonian simulation to new problems such as matrix inversion.

For the most part, this capability should be used by "advanced" users, who are interested in experimenting with QSP. The methods used in the QSP example notebooks should be sufficient for most users.

**NOTE:** This is still under development, and the legacy method for angle calculation has not yet been retired.

## Notebook Summaries
### _Angle_Generation_Overview.ipynb
This notebook provides a high-level overview of the `Angler`, `Expander`, and `Fitter` classes that handle all of the classical preprocessing that is needed to implement a  function $f(x)$ using a QSP sequence.

### Angle_Generation_Expander.ipynb
This notebook discusses how `Expander` can provide a polynomial approximation for a variety of functions through deidcated methods in `pyLIQTR`. Methods in `Expander` return Chebyshev expansion coefficients at a desired precision of $\epsilon_0 = $ `eps0`.  The degree $d$ of this approximation is thus set the precision target. For instance, the expansion for a Gaussian $f(x) = \exp[-\alpha x^2]$ may be generated using the `gaussian(alpha,eps0)` method.

### Angle_Generation_Fitter.ipynb
This notebook describes how one can use `Fitter`, which uses a  Remez multiple-exhange algorithm to find the best degree-$d$  polynomial approximation $\pi_d$ to an arbitrary function $f$. The notebook demonstrates the usage of the Remez algorithm, with an emphasis on the nuances that are required by QSP/QSVT.

