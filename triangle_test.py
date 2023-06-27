
from firedrake import *
import numpy as np
from src.CovarianceOTPrimalDualSolver import CovarianceOTPrimalDualSolver
from src.utils_firedrake import *
from src.utils import *

sdev = .4
alpha = 0.1
lamb = np.sqrt((sdev**2-alpha**2)*.5)

rho0 = lambda x0,x1 : 1/3 * ( exp(-0.5*(x1)**2/(alpha**2)) *exp(-0.5*(x0-2*lamb)**2/(alpha**2))/ (2*np.pi*alpha**2) + 
                              exp(-0.5*(x1-np.sqrt(3)*lamb)**2/(alpha**2)) *exp(-0.5*(x0+lamb)**2/(alpha**2))/ (2*np.pi*alpha**2) +                         
                              exp(-0.5*(x1+np.sqrt(3)*lamb)**2/(alpha**2)) *exp(-0.5*(x0+lamb)**2/(alpha**2))/ (2*np.pi*alpha**2))


rho1 = lambda x0,x1 : rho0(-x0,x1)


# Solution on square
#L = 5. 
#base_mesh = SquareMesh(40, 40, L)
#otsolver = CovarianceOTPrimalDualSolver(rho0,rho1, base_mesh = base_mesh, degX = 0, shift = (L*.5,L*.5))

# Solution on unit disk
otsolver = CovarianceOTPrimalDualSolver(rho0,rho1, refinement_level = 5, degX = 0)



otsolver.solve(.5,.5,variance = sdev**2,NmaxIter= 2000)



