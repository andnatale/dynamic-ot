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

   def __init__(self,W,f,u0,u):
        """ 
        :arg W: Mixed function space
        :arg f: Function (vector field) to be projected
        :arg u0: Function with correct boundary conditions
        :arg u: Solution        
        """
          
        v_basis = VectorSpaceBasis(constant=True)
        nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])

        sigma, phi= TrialFunctions(W)
        eta, psi = TestFunctions(W)

        a = (dot(sigma,eta)+ div(eta)*phi + div(sigma)*psi)*dx
        L = dot(f,eta)*dx
 
        bcs = [ DirichletBC(W.sub(0), u0,"on_boundary"),
                DirichletBC(W.sub(0), u0,"top"),
                DirichletBC(W.sub(0), u0,"bottom")]
      
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
          
        problem = LinearVariationalProblem(a,L, u, bcs = bcs)
        super(LinearVariationalSolver,self).__init__(problem,
                                                     nullspace = nullspace,
                                                     solver_parameters = parameters)
    


def ApplyOnDofs(method,f):
    """ Apply method on degrees of freedom of vector field f
     
    :arg method: function Rn -> Rn (n number of components of f) 
    :arg f: firedrake Vector Function
    """

    data = f.dat.data[:]
    
    data = method(*np.split(data,data.shape[1],1))
     
    f.dat.data[:] = np.concatenate(data,axis = 1) 
    






