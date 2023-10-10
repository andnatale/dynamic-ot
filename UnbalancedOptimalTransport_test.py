""" Test for dynamical optimal transport solver """

from firedrake import *
from src.UnbalancedOTPrimalDualSolver import UnbalancedOTPrimalDualSolver

sdev = 0.1
rho0 = lambda x0 : exp(-0.5*(x0-.3)**2/(sdev**2))#*exp(-0.5*(x1-.5)**2/(sdev**2))
rho1 = lambda x0 : .5*exp(-0.5*(x0-.7)**2/(sdev**2))#*exp(-0.5*(x1-.5)**2/(sdev**2))

otsolver = UnbalancedOTPrimalDualSolver(rho0,rho1)
otsolver.solve(1.,1.)







