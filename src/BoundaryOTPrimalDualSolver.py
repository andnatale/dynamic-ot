""" Primal-dual algorithm for dynamic Unbalanced Optimal Transport

"""

from firedrake import *
from .utils_firedrake import *
from .utils import *
from .BoundaryOptimalTransportProblem import BoundaryOptimalTransportProblem

class BoundaryOTPrimalDualSolver(BoundaryOptimalTransportProblem):
    """ Primal dual (PDGH) solver for dynamic transport problem """
    def __init__(self, rho0, rho1, gamma_0, gamma_1, base_mesh=None, layers = 20, degX = 0, shift = (0.,0.), unit_mass = False):

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
                      
        # Initialize mesh and variables (denisities normalized to have unit mass)
        super().__init__(rho0, rho1, gamma_0, gamma_1, base_mesh = base_mesh, layers = layers, degX = degX, 
                                                                         shift = shift, unit_mass = unit_mass)

    def solve(self,tau1,tau2, tol = 10e-7, NmaxIter= 2e3, projection_bulk = projection,
                                                                    projection_interface = projection):
        """
        OT problem solver 

        :arg tau1, tau2: parameters PDGH aglorithm
        :arg tol: tolerance on increment
        :arg NmaxIter: int maximum numbert iterations
        :arg projection: function proximal operator of kinetic energy (|m|^2/(2rho) if not provided) 
        """ 
        i = 0 
        err =  1.
        self.err_vec = []
        errdiv_vec = []    
        
        # Auxiliary functions bulk
        sigma_oldX = Function(self.ot_bulk.X) 
        pxi = Function(self.ot_bulk.X)

        # Auxiliary functions interface (fluxes)
        sigma_f_oldX = Function(self.ot_interface.X)
        alpha_f_old = Function(self.ot_interface.F)
        pxi_f = Function(self.ot_interface.X)
        pzeta_f = Function(self.ot_interface.F)

        # Auxiliary function (interface to boundary)
        #alpha_bulk = Function(self.ot_bulk.Fluxes) 
        #mutliplier_fluxes_bulk = Function(self.ot_bulk.Fluxes)  
        fluxes_f_old = Function(self.ot_interface.F)

        # Continuity constraint projection
        divsolver_bulk = BoundaryDivProjectorSolver(self.ot_bulk.sigmaX - tau1*self.ot_bulk.q,
                                                    self.ot_bulk.fluxes - tau1*self.ot_bulk.multiplier_fluxes,
                                                    self.ot_bulk.sigma0, 
                                                    mixed_problem = False, V= None)
        
        divsolver_interface = UnbalancedDivProjectorSolver(self.ot_interface.sigmaX -tau1*self.ot_interface.q, 
                                                           self.ot_interface.alpha-tau1*(self.ot_interface.r
                                                             + self.ot_interface.multiplier_fluxes),
                                                           self.ot_interface.sigma0, 
                                                           mixed_problem = False, V= None)
                                                                                                               
        while err > tol and i < NmaxIter:

            sigma_oldX.assign(self.ot_bulk.sigmaX)
            
            fluxes_f_old.assign(self.ot_interface.fluxes)
            sigma_f_oldX.assign(self.ot_interface.sigmaX)
            alpha_f_old.assign(self.ot_interface.alpha)
             
            
            # Proximal operator continuity bulk
            sigma_sol, fluxes_sol = divsolver_bulk.get_projected_solution(self.ot_bulk.X,self.ot_bulk.Fluxes)
            self.ot_bulk.sigmaX.assign(sigma_sol)
            self.ot_bulk.fluxes.assign(fluxes_sol)

            self.ot_interface.fluxes.dat.data[:] = self.apply_map_boundary_mesh(self.ot_bulk.fluxes)
            
            # Proximal operator continuity interface
            sigma_f_sol, alpha_f_sol = divsolver_interface.get_projected_solution(self.ot_interface.X,
                                                                                      self.ot_interface.F)
            self.ot_interface.sigmaX.assign(sigma_f_sol)
            self.ot_interface.alpha.assign(alpha_f_sol)
            
             
            # Proximal operator kinetic energy (bulk)
            pxi.assign(assemble(self.ot_bulk.q + tau2*(2*self.ot_bulk.sigmaX-sigma_oldX))) 
            ApplyOnDofsList(projection_bulk,[pxi]) 
            self.ot_bulk.q.assign(pxi)
             
            # Proximal operator kinetic energy (interface)
            pxi_f.assign(assemble(self.ot_interface.q+tau2*(2*self.ot_interface.sigmaX-sigma_f_oldX))) 
            pzeta_f.assign(assemble(self.ot_interface.r + tau2*(2*self.ot_interface.alpha - alpha_f_old)))
            ApplyOnDofsList(projection_interface,[pzeta_f,pxi_f])
            self.ot_interface.q.assign(pxi_f)
            self.ot_interface.r.assign(pzeta_f)
             
            # Update fluxes multiplier
            self.ot_interface.multiplier_fluxes.assign(self.ot_interface.multiplier_fluxes + 
                   tau2*(2*self.ot_interface.fluxes - fluxes_f_old + 2*self.ot_interface.alpha - alpha_f_old)) 
            self.ot_bulk.multiplier_fluxes.dat.data[:] = self.apply_adjoint_map_boundary_mesh(self.ot_interface.multiplier_fluxes)  
            
            # Errors
            err = np.sqrt(assemble(dot(self.ot_bulk.sigmaX-sigma_oldX,self.ot_bulk.sigmaX-sigma_oldX)*dx)) 
            self.err_vec.append(err) 
            #errdiv_vec.append(np.sqrt(assemble(div(sigma)**2*dx)))          
          
            print('Iteration  '+str(i))
            print('Optimality error : '+str(err) + 
                  ' Min density (bulk): '
                    + str(np.min(self.ot_bulk.sigmaX.dat.data[:,self.ot_bulk.time_index]))+
                  ' Min density (interface): '
                    + str(np.min(self.ot_interface.sigmaX.dat.data[:,self.ot_interface.time_index])))

            #print('Optimality error continuity : '+ str(errdiv_vec[i]))
            err = 10 
            # Update iteration counter
            i+=1
            


        # Get H(div) solution (to be modified)
        divsolver = BoundaryDivProjectorSolver(self.ot_bulk.sigmaX, self.ot_bulk.fluxes,
                                                 self.ot_bulk.sigma0, mixed_problem = False, V = None)
        sigma_sol, fluxes_sol = divsolver.get_projected_solution(self.ot_bulk.V,self.ot_bulk.Fluxes)
        self.ot_solver.sigma.assign(sigma_sol)
