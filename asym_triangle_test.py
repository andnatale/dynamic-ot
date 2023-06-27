
from firedrake import *
import numpy as np
from src.CovarianceOTPrimalDualSolver import CovarianceOTPrimalDualSolver
from src.utils_firedrake import *
from src.utils import *

#sdev = .4
alpha = 0.1

m1 = .2
m2 = (1-m1)/2
delta1 = .5
delta2 = m1/2/m2*delta1

h = np.sqrt(.5*(m1/m2 + m1**2/m2**2*.5))*delta1
sdev = np.sqrt(2*m2*h**2 + (m1+2*m2)*alpha**2)
sdev = np.sqrt(m1*delta1**2 +2*m2*delta2**2 +  (m1+2*m2)*alpha**2)


rho0 = lambda x0,x1 : (m1*exp(-0.5*(x1)**2/(alpha**2))*exp(-0.5*(x0-delta1)**2/(alpha**2))/(2*np.pi*alpha**2)
                    + m2*exp(-0.5*(x1-h)**2/(alpha**2))*exp(-0.5*(x0+delta2)**2/(alpha**2))/(2*np.pi*alpha**2)
                    + m2*exp(-0.5*(x1+h)**2/(alpha**2))*exp(-0.5*(x0+delta2)**2/(alpha**2))/(2*np.pi*alpha**2))


rho1 = lambda x0,x1 : rho0(-x0,x1)


# Solution on square
#L = 5. 
#base_mesh = SquareMesh(40, 40, L)
#otsolver = CovarianceOTPrimalDualSolver(rho0,rho1, base_mesh = base_mesh, degX = 0, shift = (L*.5,L*.5))

# Solution on unit disk
otsolver = CovarianceOTPrimalDualSolver(rho0,rho1, refinement_level = 5, layers=18, degX = 0)



otsolver.solve(.5,.5,variance = sdev**2,NmaxIter= 2000)


test_name = 'asym'
otsolver.make_gif(file_name='{}_test.gif'.format(test_name))
otsolver.make_moments_plot(orders=[[2,0],[0,2]],file_name='{}_test_moment2.png'.format(test_name))
otsolver.make_moments_plot(orders=[[4,0],[0,4]],file_name='{}_test_moment4.png'.format(test_name))




