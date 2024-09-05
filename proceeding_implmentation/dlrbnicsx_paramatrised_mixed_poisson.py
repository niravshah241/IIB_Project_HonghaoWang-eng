from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import abc
import itertools
import matplotlib.pyplot as plt
import os

import ufl
import basix
import dolfinx
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, \
    apply_lifting, set_bc

import rbnicsx
import rbnicsx.backends
import rbnicsx.online

from dlrbnicsx.neural_network.neural_network import HiddenLayersNet
from dlrbnicsx.activation_function.activation_function_factory import Tanh, Sigmoid
from dlrbnicsx.dataset.custom_dataset import CustomDataset
from dlrbnicsx.interface.wrappers import DataLoader, save_model, load_model, \
    save_checkpoint, load_checkpoint, get_optimiser, get_loss_func
from dlrbnicsx.train_validate_test.train_validate_test import \
    train_nn, validate_nn, online_nn, error_analysis

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
        sigma_h, u_h = w_h.split()
        sigma_h = sigma_h.collapse()
        u_h = u_h.collapse()
        ksp.destroy()
        A.destroy()
        L.destroy()
        return sigma_h, u_h

class PODReducedProblem:
    def __init__(self, problem):
        Q, _ = problem._V.sub(0).collapse()
        U, _ = problem._V.sub(1).collapse()
        self._basis_functions_sigma = rbnicsx.backends.FunctionsList(Q)
        self._basis_functions_u = rbnicsx.backends.FunctionsList(U)
        sigma, u = ufl.TrialFunction(Q), ufl.TrialFunction(U)
        v, q = ufl.TestFunction(Q), ufl.TestFunction(U)
        self._inner_product_sigma = ufl.inner(sigma, v) * problem.dx + \
            ufl.inner(ufl.div(sigma), ufl.div(v)) * problem.dx
        self._inner_product_action_sigma = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_sigma,
                                                  part="real")
        self._inner_product_u = ufl.inner(u, q) * problem.dx
        self._inner_product_action_u = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_u,
                                                  part="real")

        self.output_scaling_range_sigma = [-1, 1.]
        self.output_range_sigma = [None, None]
        self.output_scaling_range_u = [-1, 1.]
        self.output_range_u = [None, None]

        self.input_scaling_range = [-1, 1.]
        self.input_range = \
            np.array([[2.*np.pi, 12.],
                      [10./4.*np.pi, 13.]])
        """
        self.input_range = \
            np.array([[5, 10],
                      [5, 10]])

        self.input_range = \
            np.array([[5, 10, 50, 0.5, 0.5],
                      [5, 10, 50, 0.5, 0.5]])
        """

    def reconstruct_solution_sigma(self, reduced_solution):
        return self._basis_functions_sigma[:reduced_solution.size] * \
            reduced_solution

    def reconstruct_solution_u(self, reduced_solution):
        return self._basis_functions_u[:reduced_solution.size] * \
            reduced_solution

    def project_snapshot_sigma(self, solution, N):
        return self._project_snapshot_sigma(solution, N)

    def project_snapshot_u(self, solution, N):
        return self._project_snapshot_u(solution, N)

    def _project_snapshot_sigma(self, solution, N):
        ### This line can be replaced by an offline function?
        projected_snapshot_sigma = rbnicsx.online.create_vector(N)
        # projected_snapshot_sigma = np.random.rand(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action_sigma,
                           self._basis_functions_sigma[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action_sigma(solution),
                           self._basis_functions_sigma[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_sigma.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_sigma)
        return projected_snapshot_sigma

    def _project_snapshot_u(self, solution, N):
        projected_snapshot_u = rbnicsx.online.create_vector(N)
        # projected_snapshot_u = np.random.rand(N)
        A = rbnicsx.backends.\
            project_matrix(self._inner_product_action_u,
                           self._basis_functions_u[:N])
        F = rbnicsx.backends.\
            project_vector(self._inner_product_action_u(solution),
                           self._basis_functions_u[:N])
        ksp = PETSc.KSP()
        ksp.create(projected_snapshot_u.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.setFromOptions()
        ksp.solve(F, projected_snapshot_u)
        return projected_snapshot_u

    def update_input_range(self, input_data):
        # Find minimum and maximum value of each column
        min_values = np.min(input_data, axis=0)
        max_values = np.max(input_data, axis=0)
        result = np.stack([min_values, max_values])
        self.input_range = result

    def compute_norm_u(self, function):
        """Compute the norm of a scalar function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action_u(function)(function))

    def compute_norm_sigma(self, function):
        """Compute the norm of a flux function inner product
        on the reference domain."""
        return np.sqrt(self._inner_product_action_sigma(function)(function))

    def norm_error_u(self, u, v):
        return self.compute_norm_u(u-v)/self.compute_norm_u(u)

    def norm_error_sigma(self, sigma, q):
        return self.compute_norm_sigma(sigma-q)/self.compute_norm_sigma(sigma)

# Read mesh
# Boundary markers: 1 - Bottom, 2 - Right, 3 - Top, 4 - Left
gmsh_model_rank = 0
gdim = 2
mesh_comm = MPI.COMM_WORLD
mesh, subdomains, boundaries = dolfinx.io.gmshio.read_from_msh(
    "mesh_data/mesh.msh", mesh_comm, gmsh_model_rank, gdim=gdim)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)

online_mu = np.array([2.26 * np.pi, 12.14])
problem_parametric = ParametrisedProblem(mesh, subdomains, boundaries)
sigma_h2, u_h2 = problem_parametric.solve(online_mu)

'''
with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/sigma2.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_h2)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/u2.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_h2)
'''

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
def generate_training_set(samples=[10, 10]):
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

print("set up reduced problem")
reduced_problem = PODReducedProblem(problem_parametric)

print("")

for (mu_index, mu) in enumerate(training_set):
    print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

    print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
    print("high fidelity solve for mu =", mu)
    snapshot_sigma, snapshot_u = problem_parametric.solve(mu)

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
reduced_problem._basis_functions_sigma.extend(modes_sigma)
reduced_size_sigma = len(reduced_problem._basis_functions_sigma)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))
print(f"Eigenvalues (sigma): {eigenvalues_sigma[:reduced_size_sigma]}")


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
reduced_problem._basis_functions_u.extend(modes_u)
reduced_size_u = len(reduced_problem._basis_functions_u)
print("")

print(rbnicsx.io.TextBox("POD-Galerkin offline phase ends", fill="="))
print(f"Eigenvalues (u): {eigenvalues_u[:reduced_size_u]}")

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

exit()

# POD Ends ###

sigma_fem_solution_online_mu, u_fem_solution_online_mu = problem_parametric.solve(online_mu)

sigma_projected_solution_online_mu = reduced_problem.reconstruct_solution_sigma(reduced_problem.project_snapshot_sigma(sigma_fem_solution_online_mu, reduced_size_sigma))
u_projected_solution_online_mu = reduced_problem.reconstruct_solution_u(reduced_problem.project_snapshot_u(u_fem_solution_online_mu, reduced_size_u))

'''
with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/sigma_projected.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_projected_solution_online_mu)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/u_projected.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_projected_solution_online_mu)
'''

# 5. ANN implementation

def generate_ann_input_set(num_samples = 200):
    """Generate an equispaced training set using numpy."""
    limits = np.array([[2 * np.pi, 2.5 * np.pi],
                       [12., 13.]])
    sampling = LHS(xlimits=limits)
    para_samples = sampling(num_samples)
    return para_samples

def generate_ann_output_set(problem, reduced_problem,
                            input_set, mode=None):
    output_set_sigma = np.zeros([input_set.shape[0],
                           len(reduced_problem._basis_functions_sigma)])
    output_set_u = np.zeros([input_set.shape[0],
                           len(reduced_problem._basis_functions_u)])
    for i in range(input_set.shape[0]):
        if mode is None:
            print(f"Parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")
        else:
            print(f"{mode} parameter number {i+1} of {input_set.shape[0]}")
            print(f"Parameter: {input_set[i,:]}")

        sigma_fem_solution, u_fem_solution = problem.solve(input_set[i, :])
        output_set_sigma[i, :] = \
            reduced_problem.project_snapshot_sigma(sigma_fem_solution,
                                                   len(reduced_problem._basis_functions_sigma)).array.astype("f")
        output_set_u[i, :] = \
            reduced_problem.project_snapshot_u(u_fem_solution,
                                               len(reduced_problem._basis_functions_u)).array.astype("f")

    return output_set_sigma, output_set_u

# Training dataset
ann_input_set = generate_ann_input_set(num_samples = 200)
np.random.shuffle(ann_input_set)
ann_output_set_sigma, ann_output_set_u = \
    generate_ann_output_set(problem_parametric, reduced_problem,
                            ann_input_set, mode="Training")

num_training_samples = int(0.7 * ann_input_set.shape[0])
num_validation_samples = ann_input_set.shape[0] - num_training_samples

reduced_problem.output_range_sigma[0] = np.min(ann_output_set_sigma)
reduced_problem.output_range_sigma[1] = np.max(ann_output_set_sigma)
reduced_problem.output_range_u[0] = np.min(ann_output_set_u)
reduced_problem.output_range_u[1] = np.max(ann_output_set_u)
# NOTE Output_range based on the computed values instead of user guess.

input_training_set = ann_input_set[:num_training_samples, :]
output_training_set_sigma = ann_output_set_sigma[:num_training_samples, :]
output_training_set_u = ann_output_set_u[:num_training_samples, :]

input_validation_set = ann_input_set[num_training_samples:, :]
output_validation_set_sigma = ann_output_set_sigma[num_training_samples:, :]
output_validation_set_u = ann_output_set_u[num_training_samples:, :]

customDataset = CustomDataset(reduced_problem, input_training_set,
                              output_training_set_sigma,
                              input_scaling_range=reduced_problem.input_scaling_range,
                              output_scaling_range=reduced_problem.output_scaling_range_sigma,
                              input_range=reduced_problem.input_range,
                              output_range=reduced_problem.output_range_sigma, verbose=False)
train_dataloader_sigma = DataLoader(customDataset, batch_size=6, shuffle=False) # shuffle=True)

customDataset = CustomDataset(reduced_problem, input_validation_set,
                              output_validation_set_sigma,
                              input_scaling_range=reduced_problem.input_scaling_range,
                              output_scaling_range=reduced_problem.output_scaling_range_sigma,
                              input_range=reduced_problem.input_range,
                              output_range=reduced_problem.output_range_sigma, verbose=False)
valid_dataloader_sigma = DataLoader(customDataset, batch_size=input_validation_set.shape[0], shuffle=False) # shuffle=True)

customDataset = CustomDataset(reduced_problem, input_training_set,
                              output_training_set_u,
                              input_scaling_range=reduced_problem.input_scaling_range,
                              output_scaling_range=reduced_problem.output_scaling_range_u,
                              input_range=reduced_problem.input_range,
                              output_range=reduced_problem.output_range_u, verbose=False)
train_dataloader_u = DataLoader(customDataset, batch_size=6, shuffle=False) # shuffle=True)

customDataset = CustomDataset(reduced_problem, input_validation_set,
                              output_validation_set_u,
                              input_scaling_range=reduced_problem.input_scaling_range,
                              output_scaling_range=reduced_problem.output_scaling_range_u,
                              input_range=reduced_problem.input_range,
                              output_range=reduced_problem.output_range_u, verbose=False)
valid_dataloader_u = DataLoader(customDataset, batch_size=6, shuffle=False) # shuffle=True)

# ANN Model
model_sigma = HiddenLayersNet(input_training_set.shape[1], [15, 15],
                              len(reduced_problem._basis_functions_sigma), Tanh())
model_u = HiddenLayersNet(input_training_set.shape[1], [25, 25],
                          len(reduced_problem._basis_functions_u), Tanh())

# Start of training (sigma)
path_sigma = "model_sigma.pth"
save_model(model_sigma, path_sigma)
load_model(model_sigma, path_sigma)

training_loss_sigma = list()
validation_loss_sigma = list()

max_epochs_sigma = 1000
min_validation_loss_sigma = None
start_epoch_sigma = 0
checkpoint_path_sigma = "checkpoint_sigma"
checkpoint_epoch_sigma = 10

learning_rate_sigma = 5.e-4
optimiser_sigma = get_optimiser(model_sigma, "Adam", learning_rate_sigma)
loss_func_sigma = get_loss_func("MSE", reduction="sum")

if os.path.exists(checkpoint_path_sigma):
    start_epoch_sigma, min_validation_loss_sigma = \
        load_checkpoint(checkpoint_path_sigma, model_sigma,
                        optimiser_sigma)

import time
start_time = time.time()
for epochs in range(start_epoch_sigma, max_epochs_sigma):
    if epochs > 0 and epochs % checkpoint_epoch_sigma == 0:
        save_checkpoint(checkpoint_path_sigma, epochs,
                        model_sigma, optimiser_sigma,
                        min_validation_loss_sigma)
    print(f"Epoch: {epochs+1}/{max_epochs_sigma}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_sigma,
                                     model_sigma, loss_func_sigma, optimiser_sigma)
    training_loss_sigma.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_sigma,
                                          model_sigma, loss_func_sigma)
    validation_loss_sigma.append(current_validation_loss)
    # Earlystopping criteria
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_sigma:
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_sigma = min(validation_loss_sigma)
end_time = time.time()
elapsed_time = end_time - start_time

# Start of training (u)
path_u = "model_u.pth"
save_model(model_u, path_u)
load_model(model_u, path_u)

training_loss_u = list()
validation_loss_u = list()

max_epochs_u = 1000
min_validation_loss_u = None
start_epoch_u = 0
checkpoint_path_u = "checkpoint_u"
checkpoint_epoch_u = 10

learning_rate_u = 5.e-4
optimiser_u = get_optimiser(model_u, "Adam", learning_rate_u)
loss_func_u = get_loss_func("MSE", reduction="sum")

if os.path.exists(checkpoint_path_u):
    start_epoch_u, min_validation_loss_u = \
        load_checkpoint(checkpoint_path_u, model_u,
                        optimiser_u)

import time
start_time = time.time()
for epochs in range(start_epoch_u, max_epochs_u):
    if epochs > 0 and epochs % checkpoint_epoch_u == 0:
        save_checkpoint(checkpoint_path_u, epochs,
                        model_u, optimiser_u,
                        min_validation_loss_u)
    print(f"Epoch: {epochs+1}/{max_epochs_u}")
    current_training_loss = train_nn(reduced_problem, train_dataloader_u,
                                     model_u, loss_func_u, optimiser_u)
    training_loss_u.append(current_training_loss)
    current_validation_loss = validate_nn(reduced_problem, valid_dataloader_u,
                                          model_u, loss_func_u)
    validation_loss_u.append(current_validation_loss)
    # Earlystopping criteria
    if epochs > 0 and current_validation_loss > 1.01 * min_validation_loss_u:
        # 1% safety margin against min_validation_loss
        # before invoking early stopping criteria
        print(f"Early stopping criteria invoked at epoch: {epochs+1}")
        break
    min_validation_loss_u = min(validation_loss_u)
end_time = time.time()
elapsed_time = end_time - start_time

os.system(f"rm {checkpoint_path_sigma}")
os.system(f"rm {checkpoint_path_u}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset for sigma")
print("\n")
error_analysis_samples_sigma = 100
error_analysis_set_sigma = generate_ann_input_set(num_samples = error_analysis_samples_sigma)
error_numpy_sigma = np.zeros(error_analysis_set_sigma.shape[0])

for i in range(error_analysis_set_sigma.shape[0]):
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_sigma.shape[0]}: {error_analysis_set_sigma[i,:]}")
    error_numpy_sigma[i] = error_analysis(reduced_problem, problem_parametric,
                                          error_analysis_set_sigma[i, :], model_sigma,
                                          len(reduced_problem._basis_functions_sigma), online_nn,
                                          norm_error=reduced_problem.norm_error_sigma,
                                          reconstruct_solution=reduced_problem.reconstruct_solution_sigma,
                                          input_scaling_range=reduced_problem.input_scaling_range,
                                          output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                          input_range=reduced_problem.input_range,
                                          output_range=reduced_problem.output_range_sigma,
                                          index=0, verbose=True)
    print(f"Error: {error_numpy_sigma[i]}")

# Error analysis dataset
print("\n")
print("Generating error analysis (only input/parameters) dataset for u")
print("\n")
error_analysis_samples_u = 100
error_analysis_set_u = generate_ann_input_set(num_samples = error_analysis_samples_u)
error_numpy_u = np.zeros(error_analysis_set_u.shape[0])

for i in range(error_analysis_set_u.shape[0]):
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set_u.shape[0]}: {error_analysis_set_u[i,:]}")
    error_numpy_u[i] = error_analysis(reduced_problem, problem_parametric,
                                      error_analysis_set_u[i, :], model_u,
                                      len(reduced_problem._basis_functions_u), online_nn,
                                      norm_error=reduced_problem.norm_error_u,
                                      reconstruct_solution=reduced_problem.reconstruct_solution_u,
                                      input_scaling_range=reduced_problem.input_scaling_range,
                                      output_scaling_range=reduced_problem.output_scaling_range_u,
                                      input_range=reduced_problem.input_range,
                                      output_range=reduced_problem.output_range_u,
                                      index=1, verbose=True)
    print(f"Error: {error_numpy_u[i]}")

# Online phase at parameter online_mu
online_mu = np.array([2.47 * np.pi, 12.27])
sigma_fem_solution_online_mu, u_fem_solution_online_mu = \
    problem_parametric.solve(online_mu)
sigma_projected_solution_online_mu = \
    reduced_problem.reconstruct_solution_sigma\
        (reduced_problem.project_snapshot_sigma
         (sigma_fem_solution_online_mu, reduced_size_sigma))
u_projected_solution_online_mu = \
    reduced_problem.reconstruct_solution_u\
        (reduced_problem.project_snapshot_u
         (u_fem_solution_online_mu, reduced_size_u))

# Compute RB solution
rb_solution_sigma = \
    reduced_problem.reconstruct_solution_sigma(online_nn(reduced_problem,
                                                         problem_parametric,
                                                         online_mu, model_sigma,
                                                         len(reduced_problem._basis_functions_sigma),
                                                         input_scaling_range=reduced_problem.input_scaling_range,
                                                         output_scaling_range=reduced_problem.output_scaling_range_sigma,
                                                         input_range=reduced_problem.input_range,
                                                         output_range=reduced_problem.output_range_sigma))

rb_solution_u = \
    reduced_problem.reconstruct_solution_u(online_nn(reduced_problem,
                                                     problem_parametric,
                                                     online_mu, model_u,
                                                     len(reduced_problem._basis_functions_u),
                                                     input_scaling_range=reduced_problem.input_scaling_range,
                                                     output_scaling_range=reduced_problem.output_scaling_range_u,
                                                     input_range=reduced_problem.input_range,
                                                     output_range=reduced_problem.output_range_u))

solution_sigma_projection_error = dolfinx.fem.Function(problem_parametric._Q)
solution_u_projection_error = dolfinx.fem.Function(problem_parametric._U)

solution_sigma_projection_error.x.array[:] = abs(sigma_fem_solution_online_mu.x.array - sigma_projected_solution_online_mu.x.array)
solution_u_projection_error.x.array[:] = abs(u_fem_solution_online_mu.x.array - u_projected_solution_online_mu.x.array)

solution_sigma_error = dolfinx.fem.Function(problem_parametric._Q)
solution_u_error = dolfinx.fem.Function(problem_parametric._U)

solution_sigma_error.x.array[:] = abs(sigma_fem_solution_online_mu.x.array - rb_solution_sigma.x.array)
solution_u_error.x.array[:] = abs(u_fem_solution_online_mu.x.array - rb_solution_u.x.array)

div_error_sigma_fem = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                          (dolfinx.fem.form(ufl.inner(ufl.div(sigma_fem_solution_online_mu), ufl.div(sigma_fem_solution_online_mu)) * problem_parametric.dx)),
                          op=MPI.SUM)

div_error_sigma_projected = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                          (dolfinx.fem.form(ufl.inner(ufl.div(sigma_projected_solution_online_mu), ufl.div(sigma_projected_solution_online_mu)) * problem_parametric.dx)),
                          op=MPI.SUM)

div_error_sigma_rb = mesh.comm.allreduce(dolfinx.fem.assemble_scalar
                          (dolfinx.fem.form(ufl.inner(ufl.div(sigma_projected_solution_online_mu), ufl.div(sigma_projected_solution_online_mu)) * problem_parametric.dx)),
                          op=MPI.SUM)

print(div_error_sigma_fem, div_error_sigma_projected, div_error_sigma_rb)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/sigma_fem_online_mu.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_fem_solution_online_mu)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/u_fem_online_mu.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_fem_solution_online_mu)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/sigma_projected_online_mu.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(sigma_projected_solution_online_mu)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/u_projected_online_mu.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(u_projected_solution_online_mu)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/sigma_rb_online_mu.xdmf",
                        "w") as sigma_solution_file:
    sigma_solution_file.write_mesh(mesh)
    sigma_solution_file.write_function(rb_solution_sigma)

with dolfinx.io.XDMFFile(mesh.comm, "dlrbnicsx_mixed_poisson/u_rb_online_mu.xdmf",
                        "w") as u_solution_file:
    u_solution_file.write_mesh(mesh)
    u_solution_file.write_function(rb_solution_u)
