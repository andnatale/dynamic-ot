""" Primal-dual algorithm for dynamic Unbalanced Optimal Transport

"""

from firedrake import *
from .utils_firedrake import *
from .utils import *
from .UnbalancedOptimalTransportProblem import UnbalancedOptimalTransportProblem

class UnbalancedOTPrimalDualSolver(UnbalancedOptimalTransportProblem):
    """ Primal dual (PDGH) solver for dynamic transport problem """
  
    def __init__(self, rho0, rho1, base_mesh = None, quads = False, layers = 20, degX = 0, unit_mass= False):
        """
        :arg rho0: function, initial density
        :arg rho1: function, final density
        :arg base_mesh: space mesh
        :arg quads: flag for quadrilateral mesh if base_mesh not provided
        :arg layers: int number of time steps  
        :arg degX: int polynomial degree in space of q dual variable to (rho,m)
        """
                  
        if base_mesh is None:
            #base_mesh = UnitSquareMesh(layers,layers,quadrilateral = quads)
            base_mesh = UnitIntervalMesh(layers)
                       
        # Initialize mesh and variables (denisities normalized to have unit mass if unit_mass=True)
        super().__init__(rho0, rho1, base_mesh , layers, degX, unit_mass = unit_mass)


    def solve(self,tau1,tau2, tol = 10e-7, NmaxIter= 2e3, projection = projection):
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
        alpha_old = Function(self.F)
        pxi = Function(self.X)
        pzeta = Function(self.F)        
 
        # Continuity constraint projection
        divsolver = UnbalancedDivProjectorSolver(self.sigmaX -tau1*self.q, self.alpha-tau1*self.r,
                                                 self.sigmaX, self.alpha,
                                                 self.sigma0, 
                                                 mixed_problem = False, V= None)
                                                                               
        while err > tol and i < NmaxIter:

            sigma_oldX.assign(self.sigmaX)
            alpha_old.assign(self.alpha)

            # Proximal operator continuity
            divsolver.project()
            
            # Proximal operator kinetic energy
            pxi.assign(assemble(self.q+tau2*(2*self.sigmaX-sigma_oldX))) 
            pzeta.assign(assemble(self.r + tau2*(2*self.alpha - alpha_old)))
            ApplyOnDofsList(projection,[pzeta,pxi]) # WARNING: "projection" works if pxi has 2 components (1d)
            self.q.assign(pxi)
            self.r.assign(pzeta)
 
            # Errors
            err = np.sqrt(assemble(dot(self.sigmaX-sigma_oldX,self.sigmaX-sigma_oldX)*dx)
                          +assemble(dot(self.alpha-alpha_old,self.alpha-alpha_old)*dx)) 
            err_vec.append(err) 
            #errdiv_vec.append(np.sqrt(assemble(div(sigma)**2*dx)))          
          
            print('Iteration  '+str(i))
            print('Optimality error : '+str(err) +' Min density: '+ str(np.min(self.sigmaX.dat.data[:,self.time_index])))
            #print('Optimality error continuity : '+ str(errdiv_vec[i]))
            
            # Update iteration counter
            i+=1


        # Get H(div) solution 
        divsolver = DivProjectorSolver(self.sigmaX, self.sigma, self.sigma0, mixed_problem = True, V = self.V)
        divsolver.project()

