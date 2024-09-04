from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import abc
import itertools
import matplotlib.pyplot as plt

import ufl
import basix
import dolfinx
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, \
    apply_lifting, set_bc

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

class ParametrisedProblem(abc.ABC):
    def __init__(self, mesh, subdomains, boundaries):
        self._mesh = mesh
        self._subdomains = subdomains
        self._boundaries = boundaries
        pol_degree = 1
        Q_el = basix.ufl.element("BDMCF", mesh.basix_cell(), pol_degree)
        P_el = basix.ufl.element("DG", mesh.basix_cell(), pol_degree - 1)
        V_el = basix.ufl.mixed_element([Q_el, P_el])
        self._V = dolfinx.fem.FunctionSpace(mesh, V_el)
        self._V0 = self._V.sub(0)
        self._Q, _ = self._V0.collapse()
        self._U, _ = self._V.sub(1).collapse()
        self._trial_sigma, self._trial_u = ufl.TrialFunctions(self._V)
        self._test_sigma, self._test_u = ufl.TestFunctions(self._V)
        self._solution = dolfinx.fem.Function(self._V)
        self._x = ufl.SpatialCoordinate(mesh)
        self.dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundaries)
        self.mu_para = \
            [dolfinx.fem.Constant(mesh, PETSc.ScalarType(2.37 * np.pi)),
             dolfinx.fem.Constant(mesh, PETSc.ScalarType(12.23))]
        sigma, tau = ufl.TrialFunction(self._Q), ufl.TestFunction(self._Q)
        self._inner_product_sigma = ufl.inner(sigma, tau) * self.dx + \
            ufl.inner(ufl.div(sigma), ufl.div(tau)) * self.dx
        self._inner_product_action_sigma = \
            rbnicsx.backends.bilinear_form_action(
                self._inner_product_sigma, part="real")
        u, v = ufl.TrialFunction(self._U), ufl.TestFunction(self._U)
        self._inner_product_u = ufl.inner(u, v) * self.dx
        self._inner_product_action_u = \
            rbnicsx.backends.bilinear_form_action(
                self._inner_product_u, part="real")

    def assemble_bcs(self):
        dofs_bottom_bc = \
            dolfinx.fem.locate_dofs_topological((self._V0, self._Q),
                                                gdim-1,
                                                boundaries.find(1))
        dofs_top_bc = \
            dolfinx.fem.locate_dofs_topological((self._V0, self._Q),
                                                gdim-1,
                                                boundaries.find(3))
        bc_func_bottom = dolfinx.fem.Function(self._Q)
        bc_func_bottom.interpolate(self.f2)
        bc_func_top = dolfinx.fem.Function(self._Q)
        bc_func_top.interpolate(self.f1)
        bc_bottom = dolfinx.fem.dirichletbc(bc_func_bottom,
                                            dofs_bottom_bc,
                                            self._V0)
        bc_top = dolfinx.fem.dirichletbc(bc_func_top,
                                         dofs_top_bc,
                                         self._V0)
        return [bc_bottom, bc_top]

    def f1(self, x):
        values = np.zeros((2, x.shape[1]))
        values[1, :] = np.sin(self.mu_para[0].value * x[0])
        return values

    def f2(self, x):
        values = np.zeros((2, x.shape[1]))
        values[1, :] = - np.sin(self.mu_para[0].value * x[0])
        return values

    def linear_form(self):
        f = self.mu_para[1].value * ufl.exp(- 5 * ((self._x[0] - 0.5) *
                                                   (self._x[0] - 0.5) +
                                                   (self._x[1] - 0.5) *
                                                   (self._x[1] - 0.5)))
        L = - ufl.inner(f, self._test_u) * self.dx
        return dolfinx.fem.form(L)

    def bilinear_form(self):
        a = ufl.inner(self._trial_sigma, self._test_sigma) * self.dx + \
            ufl.inner(self._trial_u, ufl.div(self._test_sigma)) * self.dx + \
            ufl.inner(ufl.div(self._trial_sigma), self._test_u) * self.dx
        return dolfinx.fem.form(a)

    def solve(self, mu):
        self.mu_para[0].value = mu[0]
        self.mu_para[1].value = mu[1]
        bcs = self.assemble_bcs()
        a_cpp = self.bilinear_form()
        l_cpp = self.linear_form()
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
        w_h = dolfinx.fem.Function(self._V)
        ksp.solve(L, w_h.vector)
        w_h.x.scatter_forward()
        ksp.destroy()
        A.destroy()
        L.destroy()
        return w_h

# Read mesh
# Boundary markers: 1 - Bottom, 2 - Right, 3 - Top, 4 - Left
gmsh_model_rank = 0
gdim = 2
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)

mu = np.array([2.26 * np.pi, 12.14])
problem_parametric = ParametrisedProblem(mesh, subdomains, boundaries)
w_h2 = problem_parametric.solve(mu)

sigma_h2, u_h2 = w_h2.split()
sigma_h2 = sigma_h2.collapse()
u_h2 = u_h2.collapse()

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/sigma2.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_h2)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/u2.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_h2)

print(u_h2.x.array)
print(sigma_h2.x.array)

norm_u2 = \
    mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                        (dolfinx.fem.form(ufl.inner(u_h2, u_h2) * dx)),
                        op=MPI.SUM)

norm_sigma2 = \
    mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                        (dolfinx.fem.form(ufl.inner
                                          (sigma_h2,
                                           sigma_h2) * dx +
                                          ufl.inner
                                          (ufl.div(sigma_h2),
                                           ufl.div(sigma_h2)) * dx)),
                        op=MPI.SUM)

print(norm_u2)
print(norm_sigma2)

# POD Starts ###
def generate_training_set(samples=[4, 3]):
    training_set_0 = np.linspace(2. * np.pi, 2.5 * np.pi, samples[0])
    training_set_1 = np.linspace(12., 13., samples[1])
    training_set = np.array(list(itertools.product(training_set_0,
                                                   training_set_1)))
    return training_set

# TODO correct this from rbnicsx.io.on_rank_zero
training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)

Nmax_sigma, Nmax_u = 30, 20

print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
print("")

print("set up snapshots matrix")
snapshots_matrix_sigma = rbnicsx.backends.FunctionsList(problem_parametric._Q)
snapshots_matrix_u = rbnicsx.backends.FunctionsList(problem_parametric._U)

# print("set up reduced problem")
# reduced_problem = PODANNReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot = problem_parametric.solve(mu)
    snapshot_sigma, snapshot_u = snapshot.split()
    snapshot_sigma = snapshot_sigma.collapse()
    snapshot_u = snapshot_u.collapse()

    print("update snapshots matrix")
    snapshots_matrix_sigma.append(snapshot_sigma)
    snapshots_matrix_u.append(snapshot_u)

    print("")

print(rbnicsx.io.TextLine("Perform POD (sigma)", fill="#"))
eigenvalues_sigma, modes_sigma, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_sigma,
                                    problem_parametric._inner_product_action_sigma,
                                    N=Nmax_sigma, tol=1.e-6)
# reduced_problem._basis_functions.extend(modes_sigma)
# reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))
print(f"Eigenvalues (sigma): {eigenvalues_sigma}")


positive_eigenvalues_sigma = np.where(eigenvalues_sigma > 0.,
                                      eigenvalues_sigma, np.nan)
singular_values_sigma = np.sqrt(positive_eigenvalues_sigma)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

'''
for x, y in enumerate(eigenvalues[:len(reduced_problem._basis_functions)]):
    yval.append(y)
    xint.append(x+1)
'''

for x, y in enumerate(eigenvalues_sigma):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
plt.savefig("eigenvalues_sigma.png")
# plt.show()

print(rbnicsx.io.TextLine("Perform POD (u)", fill="#"))
eigenvalues_u, modes_u, _ = \
    rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    problem_parametric._inner_product_action_u,
                                    N=Nmax_u, tol=1.e-6)
# reduced_problem._basis_functions.extend(modes_sigma)
# reduced_size = len(reduced_problem._basis_functions)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))
print(f"Eigenvalues (u): {eigenvalues_u}")

positive_eigenvalues_u = np.where(eigenvalues_u > 0.,
                                  eigenvalues_u, np.nan)
singular_values_u = np.sqrt(positive_eigenvalues_u)

plt.figure(figsize=[8, 10])
xint = list()
yval = list()

'''
for x, y in enumerate(eigenvalues[:len(reduced_problem._basis_functions)]):
    yval.append(y)
    xint.append(x+1)
'''

for x, y in enumerate(eigenvalues_u):
    yval.append(y)
    xint.append(x+1)

plt.plot(xint, yval, "*-", color="orange")
plt.xlabel("Eigenvalue number", fontsize=18)
plt.ylabel("Eigenvalue", fontsize=18)
plt.xticks(xint)
plt.yscale("log")
plt.title("Eigenvalue decay", fontsize=24)
plt.tight_layout()
plt.savefig("eigenvalues_sigma.png")
# plt.show()

# POD Ends ###

