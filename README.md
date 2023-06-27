# Mixed FEM Dynamical Optimal Transport Toolbox

Finite element discretization of dynamical optimal transport problems using [Firedrake](https://www.firedrakeproject.org/). 

Contains code for simulations from:

A. Natale, and G. Todeschi. [*"A mixed finite element discretization of dynamical optimal transport."*](https://link.springer.com/article/10.1007/s10915-022-01821-y)
Journal of Scientific Computing 91.2 (2022): 38.

## Source codes

* *OTPrimalDualSolver*: class for solving dynamical OT via primal dual optimization 
* *CovarianceOTPrimalDualSolver*: class for solving dynamical OT via primal dual optimization with constained covariance
* *utils_firedrake*: finite element solvers 
* *utils*: Legendre dual of cost function

## Dependencies
Th e code is based on the finite element software [Firedrake](https://www.firedrakeproject.org/). On an Ubuntu workstation this can be installed via
```
    $ curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install
    $ python3 firedrake-install
```
More options/information on the installation and further dependencies can be found [here](https://www.firedrakeproject.org/download.html).



 
