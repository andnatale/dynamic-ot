""" Primal-dual algorithm for dynamic Optimal Transport

"""
#import sys
#sys.path.append('..')
#sys.path.append('../src')
from firedrake import *
from .utils_firedrake import *
from .utils import *
from .ExtrudedVectorField import ExtrudedVectorField
import numpy as np

# For visualization
import matplotlib.pyplot as plt


class CovarianceOTPrimalDualSolver(ExtrudedVectorField):
    """ Primal dual (PDGH) solver for dynamic transport problem with covariance constraint """
  
    def __init__(self, rho0, rho1, base_mesh = None, refinement_level = 5, layers = 15, degX = 0, shift = (0.,0.)):
        """
        :arg rho0: function, initial density
        :arg rho1: function, final density
        :arg base_mesh: space mesh
        :arg refinement_level: refinement if base_mesh not provided
        :arg layers: int number of time steps  
        :arg degX: int polynomial degree in space of q dual variable to (rho,m)
        :arg shift: shift on coordinates for moment computations
        """
        

                  
        if base_mesh is None:
            base_mesh = UnitDiskMesh(refinement_level = refinement_level)
        self.base_mesh = base_mesh
        mesh = ExtrudedMesh(base_mesh, layers, layer_height=1./layers, extrusion_type='uniform')     
        self.mesh = mesh

        # Function spaces
        F = FunctionSpace(mesh,"DG",0)
        cell, _ = F.ufl_element().cell()._cells
        
        hspace = "RT"

        DG = FiniteElement("DG", cell, 0)
        CG = FiniteElement("CG", interval, 1)
        Vv_tr_element = TensorProductElement(DG, CG)

        DGv = FiniteElement("DG", interval, 0)
        Bh = FiniteElement(hspace, cell, 1)
        Vh_tr_element = TensorProductElement(Bh,DGv)

        V_element = HDivElement(Vh_tr_element) + HDivElement(Vv_tr_element)
        V = FunctionSpace(mesh, V_element)
        F_element = TensorProductElement(FiniteElement("DG",cell,degX),DGv)

        self.X = VectorFunctionSpace(mesh, F_element)
        self.W =  V * F


        #Area of base mesh
        self.area = assemble(project(Constant(1),F)*dx)

        # Variables
        sigma = Function(V)
        super().__init__(sigma)
        self.q = Function(self.X)
       
        self.l = [Function(F) for i in range(5)]

        x = SpatialCoordinate(mesh)
        
        self.e1 = Constant(as_vector([0,0,1])) 
        self.shift= shift  
        x0,x1 = x[0] - shift[0],x[1]-shift[1]
        rho1f=  project(rho1(x0,x1)*self.e1,V)
        rho0f=  project(rho0(x0,x1)*self.e1,V)
        rho1f = rho1f / assemble(rho1f[2]*dx)
        rho0f = rho0f / assemble(rho0f[2]*dx)
        self.sigma0 = project((rho1f*x[2]+rho0f*(1.-x[2])),V)

        # Weights for mean and covariance constraints
        self.weights = [project(x0,F), 
                        project(x1,F),
                        project(x0*x1,F),
                        project(x0**2,F), 
                        project(x1**2,F)]
        
        # Error vectors for primal dual algorithm
        self.err_vec = []
        self.errdiv_vec = []    

     

    def solve(self,tau1,tau2, tol = 1e-6, NmaxIter= 1000, projection = projection,variance = 1.):
        """
        OT problem solver 

        :arg tau1, tau2: parameters PDGH aglorithm
        :arg tol: tolerance on increment
        :arg NmaxIter: int maximum numbert iterations
        :arg projection: function proximal operator of kinetic energy (|m|^2/(2rho) if not provided) 
        """ 
        j = 0 
        err =  1.
        # Auxiliary functions
        sigma_oldX = Function(self.X)
        sigmaX = Function(self.X)
        pxi = Function(self.X)
        u = Function(self.W)

        int_rho = [ ExtrudedDGFunction(self.mesh) for i in range(5)]

        # Continuity constraint projection
        sigma_new =  self.sigma - tau1*self.q -tau1*self.e1*(self.l[0]*self.weights[0]
                                               + self.l[1]*self.weights[1]
                                               + self.l[2]*self.weights[2]
                                               + self.l[3]*self.weights[3]
                                               + self.l[4]*self.weights[4])
 
        divsolver = DivProjectorSolver(self.W, sigma_new, self.sigma0, u)

        while err > tol and j < NmaxIter:

            sigma_oldX.assign(sigmaX)
            
            # Proximal operator continuity
            divsolver.solve()
            sigma, _ =  u.split()
            self.sigma.assign(sigma)
            sigmaX.assign(project(sigma,self.X))   
         
            # Proximal operator kinetic energy
            pxi.assign(assemble(self.q+tau2*(2*sigmaX-sigma_oldX)))            
            ApplyOnDofs(projection,pxi)
            self.q.assign(pxi)
 
            # Proximal operator multipliers for covariance constraint
            sigma_extraX = assemble(2*sigmaX - sigma_oldX)
            for i in range(5):
                 int_rho[i].interpolate((2*sigmaX-sigma_oldX)[2]*self.weights[i])
                 if i>=3: int_rho[i].dat.data[:] +=  -variance/self.area                
                 self.l[i].interpolate(self.l[i] + tau2*int_rho[i].compute_integrals()) 

            # Errors
            err = np.sqrt(assemble(dot(sigmaX-sigma_oldX,sigmaX-sigma_oldX)*dx))
            self.err_vec.append(err) 
            self.errdiv_vec.append(np.sqrt(assemble(div(sigma)**2*dx)))          
          
            print('Iteration  '+str(j))
            print('Optimality error : '+str(err) +' Min density: '+ str(np.min(sigmaX.dat.data[:,2])))
            print('Optimality error continuity : '+ str(self.errdiv_vec[j]))
            
            # Update iteration counter
            j+=1

    def compute_moments(self, orders = None):
         """ Compute list of moments: each moment is the vector of moments in time

             :arg orders: list of exponents (ex. orders = [[2,0],[0,2],[1,1]] of second order)
                          if None computes moments up to order 2, orders = [[1,0],[0,1],[1,1],[2,0],[0,2]] 
         """
         if orders is None:
             weights = self.weights
             
         else:
             x = SpatialCoordinate(self.mesh)
             x0 = x[0]-self.shift[0]
             x1 = x[1]-self.shift[1]
             F = FunctionSpace(self.mesh,"DG",0)
             weights = []
             for i in range(len(orders)):
                  p , q  = orders[i]
                  weights.append(project(x0**p*x1**q,F))
                  

         nmoments = len(weights)
         moments = []
         int_rho = [ ExtrudedDGFunction(self.mesh) for i in range(nmoments)]
         for i in range(nmoments):
             int_rho[i].interpolate(self.sigma[2]*weights[i]) #Same as projection
             moment = int_rho[i].compute_integrals()
             moments.append(moment.dat.data[:int_rho[i].nvdofs])
         return moments

  
    def make_moments_plot(self, orders = [[4,0],[0,4]], file_name="moments.png"):
        

        moments = self.compute_moments(orders =orders)
        times = np.linspace(0,1,len(moments[0])+1)
        times_center = (times[:-1] + times[1:])*.5
    
        fig, ax = plt.subplots(figsize=(4,3))  # create figure & 1 axis
        colors = ['k','r','b','g']
        for i in range(len(moments)):
             ax.plot(times_center, moments[i], colors[i], linewidth = 2, 
                    label= '$k_1$ = {}, $k_2$ = {}'.format(orders[i][0],orders[i][1]))
        ax.set_title('Moments of order $k_1$, $k_2$')
        ax.set_xlabel('t')
        ax.grid('on')
    
        ax.legend()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
        plt.subplots_adjust(bottom=0.15,left = .2)
        fig.savefig(file_name)   # save the figure to file
        plt.close(fig)                                           
   
     


#sdev = 0.3

##rho0 = lambda x0,x1 : exp(-0.5*(x0)**2/(sdev**2))*exp(-0.5*(x1)**2/sdev**2)/(2*np.pi*sdev**2)
##rho1 = lambda x0,x1 : exp(-0.5*(x0)**2/(sdev**2))*exp(-0.5*(x1)**2/sdev**2)/(2*np.pi*sdev**2)

#shift = .25
#sdevx = np.sqrt(sdev**2 - shift**2)

#rho0 = lambda x0,x1 : .5*(exp(-0.5*(x1-shift)**2/(sdevx**2)) + exp(-0.5*(x1+shift)**2/(sdevx**2))) *exp(-0.5*(x0)**2/sdev**2)/(2*np.pi*sdev*sdevx)
#rho1 = lambda x0,x1 : .5*(exp(-0.5*(x0-shift)**2/(sdevx**2)) + exp(-0.5*(x0+shift)**2/(sdevx**2))) *exp(-0.5*(x1)**2/sdev**2)/(2*np.pi*sdev*sdevx)

#otsolver = CovarianceOTPrimalDualSolver(rho0,rho1,degX = 0)
#otsolver.solve(.5,.5,variance = sdev**2,NmaxIter= 2000)





