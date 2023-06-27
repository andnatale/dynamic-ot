
from firedrake import *
import numpy as np
from src.CovarianceOTPrimalDualSolver import CovarianceOTPrimalDualSolver
from src.utils_firedrake import *
from src.utils import *




alpha = 0.1
delta = .6
sdev = np.sqrt((alpha**2/2 + 7/24*delta**2))



rho0 = lambda x0,x1 : ( conditional(lt(x0, 0), 1, 0) * 
                        conditional(lt(x0, -delta), 0, 1) *
                        conditional(lt(x1, delta), 1, 0) *
                        conditional(lt(x1, -delta), 0, 1) * .5/(2*delta**2)
                        + exp ( -((x0-delta/2)**2 +(x1-delta/2)**2)/(2*alpha**2) ) /(2*pi*alpha**2)/4       
                        + exp ( -((x0-delta/2)**2 +(x1+delta/2)**2)/(2*alpha**2) ) /(2*pi*alpha**2)/4 )

rho1 = lambda x0,x1 : rho0(-x0,x1)


# Solution on square
#L = 5. 
#base_mesh = SquareMesh(40, 40, L)
#otsolver = CovarianceOTPrimalDualSolver(rho0,rho1, base_mesh = base_mesh, degX = 0, shift = (L*.5,L*.5))

# Solution on unit disk


otsolver = CovarianceOTPrimalDualSolver(rho0,rho1, refinement_level = 5, layers = 18, degX = 0)

otsolver.solve(.5,.5,variance = sdev**2,NmaxIter= 2000)

test_name = 'mixture'
otsolver.make_gif(file_name='{}_test.gif'.format(test_name))
otsolver.make_moments_plot(orders=[[2,0],[0,2]],file_name='{}_test_moment2.png'.format(test_name))
otsolver.make_moments_plot(orders=[[4,0],[0,4]],file_name='{}_test_moment4.png'.format(test_name))








