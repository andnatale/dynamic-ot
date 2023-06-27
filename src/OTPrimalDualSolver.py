""" Primal-dual algorithm for dynamic Optimal Transport

"""

from firedrake import *
from .utils_firedrake import *
from .utils import *


class OTPrimalDualSolver:
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

        mesh = ExtrudedMesh(base_mesh, layers, layer_height=1./layers, extrusion_type='uniform')

        # Function spaces
        F = FunctionSpace(mesh,"DG",0)
        cell, _ = F.ufl_element().cell()._cells

        if quads:
            hspace = 'RTCF'
        else:
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

        # Variables
        self.sigma = Function(V)
        self.q = Function(self.X)
 
        x = SpatialCoordinate(mesh)

        e1 = Constant(as_vector([0,0,1]))   
        self.sigma0 = project((rho1(x[0],x[1])*x[2]+rho0(x[0],x[1])*(1.-x[2]))*e1,V)
     
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


sdev = 0.1
rho0 = lambda x0,x1 : exp(-0.5*(x0-.3)**2/(sdev**2))*exp(-0.5*(x1)**2/(sdev**2))
rho1 = lambda x0,x1 : exp(-0.5*(x0-.7)**2/(sdev**2))*exp(-0.5*(x1)**2/(sdev**2))

otsolver = OTPrimalDualSolver(rho0,rho1)
otsolver.solve(1.,1.)









