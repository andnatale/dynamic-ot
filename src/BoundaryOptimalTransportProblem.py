""" Class collecting variables and vi visualization/post-processing utilities for boundary optimal transport problems
   
    Combines OptimalTransportProblem on bulk (interior domain) and UnbalancedOptimalTransportProblem on interfaces (boundary)
 
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
from .UnbalancedOptimalTransportProblem import UnbalancedOptimalTransportProblem

from scipy.sparse import csr_matrix


class BoundaryOptimalTransportProblem:
  
    def __init__(self, rho0, rho1, gamma_0, gamma_1, base_mesh=None, layers = 15, degX = 0, shift = (0.,0.), unit_mass = False):
        
        """
        :arg rho0: function, initial density (bulk)
        :arg rho1: function, final density (bulk)
        :arg gamma0: function, initial density (interface)
        :arg gamma1: function, final density (interface)
        :arg base_mesh: space mesh
        :arg layers: int number of time steps  
        :arg degX: int polynomial degree in space of q dual variable to (rho,m)
        :arg shift: shift on coordinates for moment computations
        :arg unit_mass: flag to normalize mass to one (both on bulk and interface)
        """

        if base_mesh is None:
            self.layers = layers
            self.base_mesh = UnitSquareMesh(layers,layers)
            self.boundary_mesh = PeriodicIntervalMesh(4*layers,4)
            self.map_boundary_mesh = self.get_map_boundary_mesh_unitsquare()

        else:
            raise(NotImplementedError)

        # Initialize mesh and variables (denisities normalized to have unit mass)
        self.ot_bulk = OptimalTransportProblem(rho0, rho1, self.base_mesh, layers, degX, unit_mass = unit_mass)                                                          #unit_mass = True imposes same mass on densities
        self.ot_bulk.Fluxes = FunctionSpace(self.ot_bulk.mesh,"CR", 1, vfamily = "DG", vdegree = 0)
        
        # Fluxes on boundary ( sigma dot n = fluxes)        
        self.ot_bulk.fluxes = Function(self.ot_bulk.Fluxes)
        #Multiplier of constraint: fluxes + alpha =0
        self.ot_bulk.multiplier_fluxes = Function(self.ot_bulk.Fluxes)
        # Representation of source term alpha as bulk function
        self.ot_bulk.alpha = Function(self.ot_bulk.Fluxes)
        

        self.ot_interface = UnbalancedOptimalTransportProblem(gamma_0,gamma_1,base_mesh = self.boundary_mesh,
                                                  layers=layers, degX=degX, unit_mass = unit_mass)

        #Multiplier of constraint: fluxes + alpha =0
        self.ot_interface.multiplier_fluxes = Function(self.ot_interface.F) 
        #Represenation of fluxes as interface function
        self.ot_interface.fluxes = Function(self.ot_interface.F)


    def apply_map_boundary_mesh(self,g):
        """ Apply map to function g with  DOFS = #FACETS in bulk"""
        return np.reshape((self.map_boundary_mesh@np.reshape(g.dat.data,(-1,self.layers))),(-1))

    def apply_adjoint_map_boundary_mesh(self,g):
        """ Apply map to function g with  DOFS = #FACETS on interface"""
        return np.reshape((self.map_boundary_mesh.transpose()@np.reshape(g.dat.data,(-1,self.layers))),(-1))


    def get_map_boundary_mesh_unitsquare(self):
        """ Builds map from bulk to boundary dofs for UnitSquareMesh  (x=[0,1], y = [0,1])
            The map returns boundary dofs with clockwise ordering (planes: x=0, y=0,x=1, y=1)

            WARNING: This function is built ad hoc (TODO for general meshes) 
        """
         
        ndofs_boundary = 4*self.layers

        F = FunctionSpace(self.base_mesh,"CR",1)
        x, y = SpatialCoordinate(self.base_mesh)

        bc0 = DirichletBC(F,1 - y + x,(1,3))
        bc1 = DirichletBC(F,y-x+3,(2,4))

        f = Function(F)
        f.dat.data[:]=100
        bc1.apply(f)
        bc0.apply(f)
        indices = np.argsort(f.dat.data)
        map_fluxes_boundarydofs = csr_matrix((np.ones(ndofs_boundary), (np.arange(ndofs_boundary),indices[:ndofs_boundary])), shape=(ndofs_boundary, len(f.dat.data)))
        return map_fluxes_boundarydofs


    #    """ Example text """ 
    #    test = Function(F)
    #    bc0 = DirichletBC(F,y-x+3,(1,2,3,4))
    #    bc0.apply(test)
    #    vect0 = map_fluxes_boundarydofs@test.dat.data
    #
    #    layers = 10
    #    mesh = ExtrudedMesh(m, layers, layer_height=1./layers, extrusion_type='uniform')
    #    x, y,z = SpatialCoordinate(mesh)
    #
    #
    #    F = FunctionSpace(mesh,"DG",0)
    #    cell, _ = F.ufl_element().cell()._cells
    #
    #    CRh = FiniteElement("CR", cell,1)
    #    DGv = FiniteElement("DG",interval,0)
    #    CR_el = TensorProductElement(CRh,DGv)
    #    CR = FunctionSpace(mesh,CR_el)
    #
    #    g = Function(CR)
    #    bc = DirichletBC(CR,y-x+3,"on_boundary")
    #    bc.apply(g)
    #    vect = np.reshape((map_fluxes_boundarydofs@np.reshape(g.dat.data,(-1,layers))),(-1))
