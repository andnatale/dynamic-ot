from firedrake import *
import numpy as np


class ExtrudedDGFunction(Function):
   def __init__(self, mesh, vfamily="DG"):

       if vfamily == "DG":
           self.nvdofs = mesh.layers - 1
           vdegree = 0
       elif vfamily == "CG":
           self.nvdofs = mesh.layers
           vdegree = 1

       self.nhdofs = mesh.num_cells()

       #volumes of cells in base mesh
       self.F_base = FunctionSpace(mesh._base_mesh,"DG",0)
       self.cell_volumes = project(CellVolume(mesh._base_mesh), self.F_base ).dat.data[:]

       F = FunctionSpace(mesh,"DG",0,vfamily = vfamily, vdegree=vdegree)
       Function.__init__(self,F)

   def compute_integrals(self):
       """ Computes (horizontal) integral in space and project to DGCG or DGDG functionspace
       Assumes uniform extrusion (numbering of elements in horizontal space is preserved after extrusion) 
       """

       integrals = self.dat.data.reshape(-1, self.nvdofs).T.dot(self.cell_volumes)
       integrals_lift = Function(self._function_space)
       integrals_lift.dat.data[:] = np.tile(integrals,(self.nhdofs,))

       return integrals_lift

   def split(self):
       """ Produces the list of functions at each time step"""

       split_fun = [Function(self.F_base) for i in range(self.nvdofs)]

       for i in range(self.nvdofs):
           split_fun[i].dat.data[:] = self.dat.data.reshape(-1, self.nvdofs)[:,i]
       return split_fun


class DivProjectorSolver(LinearVariationalSolver):
    """ Mixed Poisson Solver (projection on div free constraint)"""

    def __init__(self,f,u0, mixed_problem = True, V = None):
        """ 
        :arg f: Function (vector field) to be projected
        :arg u0: H(div) function with correct boundary conditions (fluxes)
        :arg mixed_problem: flag to solve problem in mixed formulation
        :arg V: H(div) function space to be provided for mixed_formulation
        """
        self.mixed_problem = mixed_problem  
        self.f = f
      
        mesh = u0.ufl_domain()
        if mixed_problem:
            """ Set up a mixed problem with the function space from proj_f and the lowest order DG space"""
            
            F = FunctionSpace(mesh,"DG",0)
            W = V * F

            v_basis = VectorSpaceBasis(constant=True)
            nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])

            sigma, phi= TrialFunctions(W)
            eta, psi = TestFunctions(W)

            a = (dot(sigma,eta)+ div(eta)*phi + div(sigma)*psi)*dx
            L = dot(f,eta)*dx
     
            bcs = [ DirichletBC(W.sub(0), u0, "on_boundary"),
                    DirichletBC(W.sub(0), u0, "top"),
                    DirichletBC(W.sub(0), u0, "bottom")]
          
            parameters = {"ksp_type": "gmres",
                          "ksp_rtol": 1e-12,
                          "pc_type": "fieldsplit",
                          "pc_fieldsplit_type": "schur",
                          "pc_fieldsplit_schur_fact_type": "full",
                          "fieldsplit_0_ksp_type": "preonly",
                          "fieldsplit_0_pc_type": "ilu",
                          "fieldsplit_1_ksp_type": "preonly",
                          "pc_fieldsplit_schur_precondition": "selfp",
                          "fieldsplit_1_pc_type": "hypre" }
            
            self.u = Function(W)  
            problem = LinearVariationalProblem(a,L, self.u, bcs = bcs)
        
        else:
            """ Projector on the weakly enforced div constraint """        

            n = FacetNormal(mesh)   
            W = FunctionSpace(mesh, "CR", 1, vfamily="CG", vdegree= 1)
           
            phi = TrialFunction(W)
            psi = TestFunction(W)
            
            a = dot(grad(phi),grad(psi))*dx
            L = -dot(f,grad(psi))*dx + dot(u0,n)*psi*ds_tb
        
            nullspace = VectorSpaceBasis(constant=True)
     
            parameters = {"ksp_type": "cg",
                          "pc_type": "ilu"}
            
            self.u = Function(W)  
            problem = LinearVariationalProblem(a,L, self.u)

         
        super(LinearVariationalSolver,self).__init__(problem,
                                                         nullspace = nullspace,
                                                         solver_parameters = parameters)

    def get_projected_solution(self,X):
        """ Project solution on X"""
        if self.mixed_problem:
            self.solve()
            sigma, _ =  self.u.split()
            return project(sigma, X)
        else: 
            self.solve() 
            return project(self.f+grad(self.u),X)
             
  
class UnbalancedDivProjectorSolver(LinearVariationalSolver):
    """ Mixed Poisson Solver (projection on div free constraint)"""

    def __init__(self,f,g,u0, mixed_problem = True, V = None):
        """ 
        :arg f: Function (vector field) to be projected
        :arg g: Function (scalar field) to be projected
        :arg u0: H(div) function with correct boundary conditions (fluxes)
        :arg mixed_problem: flag to solve problem in mixed formulation
        :arg V: H(div) function space to be provided for mixed_formulation
        """
        self.mixed_problem = mixed_problem  
        self.f = f
        self.g = g

        mesh = u0.ufl_domain()
        if mixed_problem:
            """ Set up a mixed problem with the function space from proj_f and the lowest order DG space"""
            raise NotImplementedError        
        else:
            """ Projector on the weakly enforced div constraint """        

            n = FacetNormal(mesh)   
            W = FunctionSpace(mesh, "CR", 1, vfamily="CG", vdegree= 1)
           
            phi = TrialFunction(W)
            psi = TestFunction(W)
            
            a = dot(grad(phi),grad(psi))*dx + phi*psi*dx
            L = -dot(f,grad(psi))*dx + dot(u0,n)*psi*ds_tb-g*psi*dx
        
            nullspace = VectorSpaceBasis(constant=True)
     
            parameters = {"ksp_type": "cg",
                          "pc_type": "ilu"}
            
            self.u = Function(W)  
            problem = LinearVariationalProblem(a,L, self.u)

         
        super(LinearVariationalSolver,self).__init__(problem,
                                                         nullspace = nullspace,
                                                         solver_parameters = parameters)

    def get_projected_solution(self,X0,X1):
        """ Project solution on X"""
        if self.mixed_problem:
            raise NotImplementedError
        else: 
            self.solve() 
            return project(self.f+grad(self.u),X0), project(self.g-self.u,X1)
             
  
   
def ApplyOnDofs(method,f,*args):
    """ Apply method on degrees of freedom of vector or scalar field f
     
    :arg method: function Rn -> Rn (n number of components of f) 
    :arg f: firedrake Vector Function
    """

    data = f.dat.data[:]
    
    if len(data.shape)>1:
        data = method(*np.split(data,data.shape[1],1),*args)
        f.dat.data[:] = np.concatenate(data,axis = 1)
    else:
        data = method(data,*args)
        f.dat.data[:] = data.copy() 
    






