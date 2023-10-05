""" Class collecting variables and vi visualization/post-processing utilities for UNBALANCED OT 
   
        Primal dual variables: density momentum
    Assumes vertical component CG1 in vertical DG0 in horizontal
"""

#import sys
#sys.path.append('..')
#sys.path.append('../src')
from firedrake import *
from .utils_firedrake import *
import numpy as np

# For visualization
import matplotlib.pyplot as plt
#import imageio
import os
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from matplotlib.ticker import FormatStrFormatter

from .OptimalTransportProblem import OptimalTransportProblem 



class UnbalancedOptimalTransportProblem(OptimalTransportProblem):
  
    def __init__(self, rho0, rho1, base_mesh , layers = 15, degX = 0, shift = (0.,0.), unit_mass = False):
        
        """
        UNBALANCED OT SETTING. Continuity equation: div(sigma) + alpha = 0
 
        :arg rho0: function, initial density
        :arg rho1: function, final density
        :arg base_mesh: space mesh
        :arg layers: int number of time steps  
        :arg degX: int polynomial degree in space of q dual variable to (rho,m)
        :arg shift: shift on coordinates for moment computations
        :arg unit_mass: flag to normalize mass to one
        """
        # Initialize mesh and variables (denisities normalized to have unit mass)
        super().__init__(rho0, rho1, base_mesh , layers, degX, unit_mass = True)

        self.alpha = Function(self.F) # Scalar function for unbalanced setting (div sigma + alpha =0)        
        self.r = Function(self.F) # Dual variable to alpha

