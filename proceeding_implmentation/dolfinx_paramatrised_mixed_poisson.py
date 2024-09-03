from mpi4py import MPI
from petsc4py import PETSc

import ufl
import basix
import dolfinx
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, \
    apply_lifting, set_bc

import numpy as np

# Import mesh in dolfinx
# Boundary markers: 1 - Bottom, 2 - Right, 3 - Top, 4 - Left
gmsh_model_rank = 0
gdim = 2
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_data/mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(mesh)
    mesh_file_xdmf.write_meshtags(subdomains, mesh.geometry)
    mesh_file_xdmf.write_meshtags(boundaries, mesh.geometry)

# Set parameter value
mu_para = [dolfinx.fem.Constant(mesh, PETSc.ScalarType(2.37 * np.pi)),
           dolfinx.fem.Constant(mesh, PETSc.ScalarType(12.23))]

# Define functions spaces, trial and test functions
pol_degree = 1
Q_el = basix.ufl.element("BDMCF", mesh.basix_cell(), pol_degree)
P_el = basix.ufl.element("DG", mesh.basix_cell(), pol_degree - 1)
V_el = basix.ufl.mixed_element([Q_el, P_el])
V = dolfinx.fem.FunctionSpace(mesh, V_el)

sigma, u = ufl.TrialFunctions(V)
tau, v = ufl.TestFunctions(V)
x = ufl.SpatialCoordinate(mesh)

V0 = V.sub(0)
Q, _ = V0.collapse()

dofs_bottom_bc = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1, boundaries.find(1))
dofs_top_bc = dolfinx.fem.locate_dofs_topological((V0, Q), gdim-1, boundaries.find(3))

def f1(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = np.sin(mu_para[0].value * x[0])
    return values

def f2(x):
    values = np.zeros((2, x.shape[1]))
    values[1, :] = - np.sin(mu_para[0].value * x[0])
    return values

f = mu_para[1].value * ufl.exp(- 5 * ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)))

bc_func_bottom = dolfinx.fem.Function(Q)
bc_func_bottom.interpolate(f2)
bc_func_top = dolfinx.fem.Function(Q)
bc_func_top.interpolate(f1)

bc_bottom = dolfinx.fem.dirichletbc(bc_func_bottom, dofs_bottom_bc, V0)
bc_top = dolfinx.fem.dirichletbc(bc_func_top, dofs_top_bc, V0)

bcs = [bc_bottom, bc_top]

a = ufl.inner(sigma, tau) * dx + ufl.inner(u, ufl.div(tau)) * dx + ufl.inner(ufl.div(sigma), v) * dx
L = - ufl.inner(f, v) * dx

problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                      "pc_factor_mat_solver_type": "mumps"})
w_h = problem.solve()
sigma_h, u_h = w_h.split()
sigma_h = sigma_h.collapse()
u_h = u_h.collapse()

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/sigma.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_h)

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/u.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_h)

a_cpp = dolfinx.fem.form(a)
l_cpp = dolfinx.fem.form(L)

A = assemble_matrix(a_cpp, bcs=bcs)
A.assemble()

L = assemble_vector(l_cpp)
apply_lifting(L, [a_cpp], [bcs])
L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(L, bcs)

ksp = PETSc.KSP()
ksp.create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.setFromOptions()
w_h2 = dolfinx.fem.Function(V)
ksp.solve(L, w_h2.vector)

sigma_h2, u_h2 = w_h2.split()
sigma_h2 = sigma_h2.collapse()
u_h2 = u_h2.collapse()

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/sigma2.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_h2)

with dolfinx.io.XDMFFile(mesh.comm, "out_mixed_poisson/u2.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_h2)

error_u = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u_h - u_h2, u_h - u_h2) * dx)) , op=MPI.SUM)

error_sigma = \
    mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(sigma_h - sigma_h2, sigma_h - sigma_h2) * dx +
                                                                     ufl.inner(ufl.div(sigma_h - sigma_h2), ufl.div(sigma_h - sigma_h2)) * dx)), op=MPI.SUM)

print(f"Error u: {error_u}, Error sigma: {error_sigma}")

print(u_h.x.array)
print(sigma_h.x.array)
print(u_h2.x.array)
print(sigma_h2.x.array)
