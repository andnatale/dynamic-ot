from firedrake import *
from utils_firedrake import ExtrudedDGFunction
import numpy as np



#m = Mesh("zmesh.msh")

m = UnitDiskMesh(refinement_level=5)
Ntimes = 10
mesh = ExtrudedMesh(m, Ntimes, layer_height=1./Ntimes, extrusion_type='uniform')


f = ExtrudedDGFunction(mesh,vfamily = "DG")




x =  SpatialCoordinate(mesh)
x0f = project(x[0]**2,f._function_space)



sdev = .2
rho = exp(-0.5*(x[0])**2/(sdev**2))*exp(-0.5*(x[1])**2/(sdev**2))/(2*np.pi*sdev**2)

g = project(rho,f._function_space)
f.interpolate(g*x0f)

integrals_f = f.compute_integrals()

output = File('test_integrals.pvd')
output.write(integrals_f)










