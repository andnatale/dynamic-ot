""" Hdiv conforming extruded vector field visualization/post-processing utilities
   
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




class ExtrudedVectorField:
  
    def __init__(self,mesh,sigma):
        

        """
        :arg sigma: Hdiv vector field on tensor product mesh  
        """
        self.mesh = mesh
        self.sigma = sigma
        

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
   

