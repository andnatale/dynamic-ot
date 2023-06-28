""" Primal-dual algorithm for dynamic Optimal Transport

"""

from firedrake import *
from .utils_firedrake import *
from .utils import *
from .OptimalTransportProblem import OptimalTransportProblem

class OTPrimalDualSolver(OptimalTransportProblem):
    """ Primal dual (PDGH) solver for dynamic transport problem """
  
    def __init__(self, rho0, rho1, base_mesh = None, quads = False, layers = 10, degX = 0):
        """
        :arg rho0: function, initial density
        :arg rho1: function, final density
        :arg base_mesh: space mesh
        :arg quads: flag for quadrilateral mesh if base_mesh not provided
        :arg layers: int number of time steps  
        :arg degX: int polynomial degree in space of q dual variable to (rho,m)
        """
                  
        if base_mesh is None:
            base_mesh = UnitSquareMesh(layers,layers,quadrilateral = quads)
            
        # Initialize mesh and variables (denisities normalized to have unit mass)
        super().__init__(rho0, rho1, base_mesh , layers, degX, unit_mass = True)


    def solve(self,tau1,tau2, tol = 10e-7, NmaxIter= 100, projection = projection):
        """
        OT problem solver 

        :arg tau1, tau2: parameters PDGH aglorithm
        :arg tol: tolerance on increment
        :arg NmaxIter: int maximum numbert iterations
        :arg projection: function proximal operator of kinetic energy (|m|^2/(2rho) if not provided) 
        """ 
        i = 0 
        err =  1.
        err_vec = []
        errdiv_vec = []    

        # Auxiliary functions
        sigma_oldX = Function(self.X)
        sigmaX = Function(self.X)
        pxi = Function(self.X)
        u = Function(self.W)

        # Continuity constraint projection
        divsolver = DivProjectorSolver(self.W, self.sigma -tau1*self.q ,self.sigma0, u)

        while err > tol and i < NmaxIter:

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
 
            # Errors
            err = np.sqrt(assemble(dot(sigmaX-sigma_oldX,sigmaX-sigma_oldX)*dx))
            err_vec.append(err) 
            errdiv_vec.append(np.sqrt(assemble(div(sigma)**2*dx)))          
          
            print('Iteration  '+str(i))
            print('Optimality error : '+str(err) +' Min density: '+ str(np.min(sigmaX.dat.data[:,2])))
            print('Optimality error continuity : '+ str(errdiv_vec[i]))
            
            # Update iteration counter
            i+=1




