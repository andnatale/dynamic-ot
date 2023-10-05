""" Class collecting variables and vi visualization/post-processing utilities for optimal transport problems
   
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




class OptimalTransportProblem:
  
    def __init__(self, rho0, rho1, base_mesh , layers = 15, degX = 0, shift = (0.,0.), unit_mass = False):
        
        """
        :arg rho0: function, initial density
        :arg rho1: function, final density
        :arg base_mesh: space mesh
        :arg layers: int number of time steps  
        :arg degX: int polynomial degree in space of q dual variable to (rho,m)
        :arg shift: shift on coordinates for moment computations
        :arg unit_mass: flag to normalize mass to one
        """

        self.base_mesh = base_mesh
        mesh = ExtrudedMesh(base_mesh, layers, layer_height=1./layers, extrusion_type='uniform')
        self.mesh = mesh

        # Function spaces
        F = FunctionSpace(mesh,"DG",0)
        cell, _ = F.ufl_element().cell()._cells

        if base_mesh.topological_dimension() ==1: hspace ="CG"
        else: hspace = "RT"

        DG = FiniteElement("DG", cell, 0)
        CG = FiniteElement("CG", interval, 1)
        Vv_tr_element = TensorProductElement(DG, CG)

        DGv = FiniteElement("DG", interval, 0)
        Bh = FiniteElement(hspace, cell, 1)
        Vh_tr_element = TensorProductElement(Bh,DGv)

        V_element = HDivElement(Vh_tr_element) + HDivElement(Vv_tr_element)
        V = FunctionSpace(mesh, V_element)
        F_element = TensorProductElement(FiniteElement("DG",cell,degX),DGv)
       
        self.F = F #DG0 space
        self.V = V #HDIV space
        self.X = VectorFunctionSpace(mesh, F_element) #DG SPACE (vector field)

        #Area of base mesh
        self.area = assemble(project(Constant(1),F)*dx)

        # Variables
        self.sigma = Function(V)
        self.q = Function(self.X)
        self.sigmaX = Function(self.X)

        x = SpatialCoordinate(mesh)
        if base_mesh.topological_dimension() ==1:
            time_index = 1
            self.e1 = Constant(as_vector([0,1]))
            rho1f=  project(rho1(x[0])*self.e1,V)
            rho0f=  project(rho0(x[0])*self.e1,V)

        elif base_mesh.topological_dimension() ==2:
            time_index = 2
            self.e1 = Constant(as_vector([0,0,1]))
            self.shift= shift
            self.x0,self.x1 = x[0] - shift[0],x[1]-shift[1]
            rho1f=  project(rho1(self.x0,self.x1)*self.e1,V)
            rho0f=  project(rho0(self.x0,self.x1)*self.e1,V)
        else:
            raise NotImplementedError
    
        if unit_mass:
            rho1f = rho1f / assemble(rho1f[time_index]*dx)
            rho0f = rho0f / assemble(rho0f[time_index]*dx)
        
        self.sigma0 = project((rho1f*x[time_index]+rho0f*(1.-x[time_index])),V)

        self.time_index = time_index

    def extract_vertical_component(self):
        """ Get list of vertical component/extruded direction (i.e., densities for OT) at different times"""
                
        rho = ExtrudedDGFunction(self.mesh,vfamily = "CG")
        rho.interpolate(self.sigma[2])
        return rho.split() 
 
    def make_gif(self,file_name = "output.gif", clean=False):


        frames = self.extract_vertical_component() 


        # Initialize a list to store individual frame filenames
        frame_filenames = []
        
        vmin = -1e-2 
        vmax = np.max([np.max(frames[i].dat.data[:]) for i in range(len(frames))])+1e-2
        n = 40  
        levels = np.linspace(vmin, vmax, n+1)
        # Generate individual frames and save them as temporary files
        for i in range(len(frames)):
     
            fig, axes = plt.subplots()
            contours = tricontourf(frames[i],levels = levels,axes=axes, cmap="coolwarm")
            axes.axis('off')
            axes.set_aspect("equal")
            frame_filename = f'frame_{i}.png'  # Generate a unique filename for each frame
            fig.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
            frame_filenames.append(frame_filename)
            
            fig.clf()
                        
        images = []
        # Open and append each image to the list
        for filename in frame_filenames:
            img = Image.open(filename)
            images.append(img)
        
        # Save the images as a GIF
        output_file = file_name
        images[0].save(output_file, save_all=True, append_images=images[1:], duration=200, loop=0)
        plt.close('all')
        if clean:
            # Clean up temporary frame files
            for frame_filename in frame_filenames:
                os.remove(frame_filename)
   

