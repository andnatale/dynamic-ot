# Primal-dual algorithm for dynamic Optimal Transport
# H1 regularization
# =========================

from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
from ALG2Tools import *

dim = 2 


# Define mesh
quads = False 
regL2 = False 
regH1 = False 
mesh_size = 10


if dim == 1:
   m = UnitIntervalMesh(mesh_size)
   # Index for vertical  coordinate
   ivert = 1 
   # Vertical unit vector  
   e1 = Constant(as_vector([0,1]))
elif dim ==2:
   m = Mesh("zmesh.msh")
   #m= PeriodicUnitSquareMesh(mesh_size,mesh_size,direction ='both',quadrilateral = quads)
   #m = UnitSquareMesh(mesh_size,mesh_size,quadrilateral = quads)
   # Index for vertical  coordinate
   ivert = 2
   # Vertical unit vector  
   e1 = Constant(as_vector([0,0,1]))
Ntimes = 30
mesh = ExtrudedMesh(m, Ntimes, layer_height=1./Ntimes, extrusion_type='uniform')



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

DGh1 = FiniteElement("DG", interval, 1)
Vh_tr_element1 = TensorProductElement(Bh,DGh1)

Q_element1 = HDivElement(Vh_tr_element1) 
Q1 = FunctionSpace(mesh, Q_element1)

F_element = TensorProductElement(FiniteElement("DG",cell,degX),DGh)
#F = FunctionSpace(mesh, F_element)

#mesh = UnitCubeMesh(mesh_size, mesh_size,mesh_size)
##mesh = UnitSquareMesh(mesh_size,mesh_size)
#Q = FunctionSpace(mesh,"RT",1)
#F = FunctionSpace(mesh,"DG",0)
#ivert = 2 
#e1 = Constant(as_vector([0,0,1]))

#X = VectorFunctionSpace(mesh,"DG",0)
X = VectorFunctionSpace(mesh, F_element)

W =  Q * Q1*  F

v_basis = VectorSpaceBasis(constant=True)
nullspace = MixedVectorSpaceBasis(W, [W.sub(0), W.sub(1), v_basis])

# Variables
sigma = Function(Q)
sigmaX = Function(X)
sigma_oldX = Function(X)
q = Function(X)

x = SpatialCoordinate(mesh)

#rho0 = (1+sign(x[0]-.5))*(1+sign(x[1]-.5))
#rho1 = (1-sign(x[0]-.5))*(1-sign(x[1]-.5))

#rho0 = exp(-(0.8-x[0])**2/0.02)*exp(-(0.2-x[1])**2/0.02) #(1+sign(x[0]-.5))*(1+sign(x[1]-.5))
#rho1 = exp(-(0.8-x[0])**2/0.02)*exp(-(0.8-x[1])**2/0.02) #(1-sign(x[0]-.5))*(1-sign(x[1]-.5))
sdev = 0.1
#rho0 = (1./(sdev**2*(2*np.pi)))*exp(-0.5*(0.65-x[0])**2/(sdev**2))*exp(-0.5*(0.35-x[1])**2/(sdev**2)) #(1+sign(x[0]-.5))*(1+sign(x[1]-.5))
#rho1 = (1./(sdev**2*(2*np.pi)))*exp(-0.5*(0.35-x[0])**2/(sdev**2))*exp(-0.5*(0.65-x[1])**2/(sdev**2)) #(1-sign(x[0]-.5))*(1-sign(x[1]-.5))
rho0 = exp(-0.5*(0.5-x[0])**2/(sdev**2))*exp(-0.5*(0.1-x[1])**2/(sdev**2)) #(1+sign(x[0]-.5))*(1+sign(x[1]-.5))
rho1 = exp(-0.5*(0.5-x[0])**2/(sdev**2))*exp(-0.5*(0.9-x[1])**2/(sdev**2)) #(1-sign(x[0]-.5))*(1-sign(x[1]-.5))

sigma0 = project((rho1*x[ivert]+rho0*(1.-x[ivert]))*e1,Q)

# Parameters
tau = 1. 
kappa = 0.002
#C_stab = mesh_size
#C_stab = project(1./(sqrt((x[0]-.3)**2+(x[1]-.3)**2)*sqrt((x[0]-.7)**2+(x[1]-.7)**2)+1e-5),F)
#kappa = C_stab*kappa
# Poisson Solver 

betat, etat, phit = TrialFunctions(W)
p, g,  psi = TestFunctions(W)

# Optimal Transport
a = (dot(betat,p) - kappa*tau*div(etat)*dot(p,e1) + tau*div(p)*phit + div(betat)*psi + dot(betat,e1)*div(g) + dot(g,etat) )*dx

L = dot(sigma-tau*q,p)*dx 

#bcs = DirichletBC(W.sub(0),sigma0,"on_boundary")
bcs = [ DirichletBC(W.sub(0), sigma0,"on_boundary"),
        DirichletBC(W.sub(0), sigma0,"top"),
        DirichletBC(W.sub(0), sigma0,"bottom"),
        DirichletBC(W.sub(1), 0,"on_boundary"),
        DirichletBC(W.sub(1), 0,"top"),
        DirichletBC(W.sub(1), 0,"bottom")]

w = Function(W)
wu,we,ws = split(w)


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


solver_parameters={"mat_type": "aij",
                            "snes_monitor": None,
                            "ksp_type": "gmres",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps"}



# Solver

err =[]
errdiv = []
errtol =1e-7
errt = 1.
i=0
#zX1 = Function(X)

while (errt>errtol or i<100)  : 
    sigma_oldX.assign(project(sigma,X))
    solve(a == L, w,nullspace = nullspace,bcs =bcs,solver_parameters = solver_parameters)
    
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
    
    alpha,beta1,beta2 = coneProjectionJDv(a0,b1,b2)
    q.dat.data[:,0] = beta1.copy()
    
    if dim ==1:
        q.dat.data[:,1] = alpha.copy()       
    elif dim ==2:
        q.dat.data[:,1] = beta2.copy()
        q.dat.data[:,2] = alpha.copy()
    
    errt = np.sqrt(assemble(dot(sigmaX-sigma_oldX,sigmaX-sigma_oldX)*dx))
    err.append(errt)
    print('Iteration  '+str(i))
    
     
    #indneg = sigma.dat.data[:,ivert]<0
    #sigma.dat.data[:,ivert][indneg] = 0
    print('Optimality error : '+str(err[i]) +' Min density: '+ str(np.min(sigmaX.dat.data[:,ivert]))) 

    errdiv.append(np.sqrt(assemble(div(sigma)**2*dx)))
    print('Optimality error div : '+ str(errdiv[i]))     
    i+=1


