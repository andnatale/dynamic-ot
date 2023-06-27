""" Primal-dual algorithm for dynamic Optimal Transport

Numerical tests from : A. Natale, and G. Todeschi. "A mixed finite element discretization of dynamical optimal transport." arXiv preprint arXiv:2003.04558 (2020).

"""

from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
from utils import *

dim = 2 

# Mesh 

# Flag for quadrialteral elements
quads = False 


mesh_size = 20

if dim == 1:
   m = UnitIntervalMesh(mesh_size)
   # Index for vertical  coordinate
   ivert = 1 
   # Vertical (time) unit vector  
   e1 = Constant(as_vector([0,1]))
elif dim ==2:
   #m = Mesh("zmesh.msh")
   #m= PeriodicUnitSquareMesh(mesh_size,mesh_size,direction ='both',quadrilateral = quads)
   #m = UnitSquareMesh(mesh_size,mesh_size,quadrilateral = quads)
   m = UnitDiskMesh(refinement_level=0) 
   # Index for vertical  coordinate
   ivert = 2
   # Vertical (time) unit vector  
   e1 = Constant(as_vector([0,0,1]))
Ntimes = 10
mesh = ExtrudedMesh(m, Ntimes, layer_height=1./Ntimes, extrusion_type='uniform') 
              



# Function spaces
degX = 1 
F = FunctionSpace(mesh,"DG",0)
cell, _ = F.ufl_element().cell()._cells

if dim == 1:
    hspace = 'CG'
elif dim ==2:
    if quads:
        hspace = 'RTCF'
    else:
        hspace = "RT"

DG = FiniteElement("DG", cell, 0)
CG = FiniteElement("CG", interval, 1)
Vv_tr_element = TensorProductElement(DG, CG)

DGh = FiniteElement("DG", interval, 0)
Bh = FiniteElement(hspace, cell, 1)
Vh_tr_element = TensorProductElement(Bh,DGh)


Q_element = HDivElement(Vh_tr_element) + HDivElement(Vv_tr_element)
Q = FunctionSpace(mesh, Q_element)
F_element = TensorProductElement(FiniteElement("DG",cell,degX),DGh)


X = VectorFunctionSpace(mesh, F_element)

W =  Q * F

v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), v_basis])

# Variables
sigma = Function(Q)
sigmaX = Function(X)
sigma_oldX = Function(X)
q = Function(X)

x = SpatialCoordinate(mesh)

sdev = 0.1
rho0 = exp(-0.5*(x[0])**2/(sdev**2))*exp(-0.5*(x[1])**2/(sdev**2))
rho1 = exp(-0.5*(x[0])**2/(sdev**2))*exp(-0.5*(x[1])**2/(sdev**2)) 

sigma0 = project((rho1*x[ivert]+rho0*(1.-x[ivert]))*e1,Q)

# Parameters
tau = 1. 
kappa = .002

betat, phit = TrialFunctions(W)
p, psi = TestFunctions(W)

# Optimal Transport
a = (dot(betat,p)+ tau*div(p)*phit + div(betat)*psi)*dx

L = dot(sigma-tau*q,p)*dx 

#bcs = DirichletBC(W.sub(0),sigma0,"on_boundary")
bcs = [ DirichletBC(W.sub(0), sigma0,"on_boundary"),
        DirichletBC(W.sub(0), sigma0,"top"),
        DirichletBC(W.sub(0), sigma0,"bottom")]


w = Function(W)
wu,ws = split(w)


parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-12,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "ilu",
    "fieldsplit_1_ksp_type": "preonly",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_1_pc_type": "hypre"
}



# Solver

err =[]
errdiv = []
errtol =1e-7
errt = 1.
i=0

while (errt>errtol ) : 
    sigma_oldX.assign(project(sigma,X))
    solve(a == L, w,nullspace = nullspace,bcs =bcs,solver_parameters = parameters)
    
    sigma.assign(project(wu,Q))
    sigmaX.assign(project(sigma,X))
      
    # Projecion on parabola
    
    pxi_data = q.dat.data + tau*(2*sigmaX.dat.data - sigma_oldX.dat.data)
    if dim ==1:
        a0 = pxi_data[:,1].copy()
        b1 = pxi_data[:,0].copy()
        b2 = np.zeros(b1.shape)
    elif dim ==2: 
        a0 = pxi_data[:,2].copy()
        b1 = pxi_data[:,0].copy()   
        b2 = pxi_data[:,1].copy()
    
    alpha,beta1,beta2 = projection(b1,b2,a0)
    q.dat.data[:,0] = beta1.copy()
    
    if dim ==1:
        q.dat.data[:,1] = alpha.copy()       
    elif dim ==2:
        q.dat.data[:,1] = beta2.copy()
        q.dat.data[:,2] = alpha.copy()
    
    errt = np.sqrt(assemble(dot(sigmaX-sigma_oldX,sigmaX-sigma_oldX)*dx))
    err.append(errt)
    print('Iteration  '+str(i))

    print('Optimality error : '+str(err[i]) +' Min density: '+ str(np.min(sigmaX.dat.data[:,ivert]))) 

    errdiv.append(np.sqrt(assemble(div(sigma)**2*dx)))
    print('Optimality error div : '+ str(errdiv[i]))     
    i+=1
    
#Citations.print_at_exit()
#output = File('Z_RT0X1L2_l1.pvd')
#output.write(sigma)
#np.save('ZerrL2_l1.npy')
    
    
