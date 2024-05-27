from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from basix.ufl import element, mixed_element
import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
from ufl import Measure, SpatialCoordinate, TestFunctions, TrialFunctions, div, exp, inner, grad, dx
import ufl
import rbnicsx
import rbnicsx.backends
import rbnicsx.io
import rbnicsx.online
import itertools
from smt.sampling_methods import LHS
from customed_dataset import CustomDataset, create_Dataloader
from ANN_model import MixedPoissonANNModel, create_model
from engine import train, online_nn, error_analysis
import tensorflow as tf
import matplotlib.pyplot as plt
import time

print(dolfinx.__version__)

class MixedPoissonSolver:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        ### TODO: Change mesh
        self.msh = mesh.create_unit_square(self.comm, 50, 50, mesh.CellType.quadrilateral)
        self.MU = [fem.Constant(self.msh, PETSc.ScalarType(5.)),
                    fem.Constant(self.msh, PETSc.ScalarType(10.)),
                    fem.Constant(self.msh, PETSc.ScalarType(1./0.02)),
                    fem.Constant(self.msh, PETSc.ScalarType(0.5)),
                    fem.Constant(self.msh, PETSc.ScalarType(0.5))]
        self.k = 1
        self.Q_el = element("BDMCF", self.msh.basix_cell(), self.k)
        self.P_el = element("DG", self.msh.basix_cell(), self.k - 1)
        self.V_el = mixed_element([self.Q_el, self.P_el])
        self.V = fem.functionspace(self.msh, self.V_el)
                
        # Define UFL variables and forms
        (self.sigma, self.u) = TrialFunctions(self.V)
        (self.tau, self.v) = TestFunctions(self.V)
        self.x = SpatialCoordinate(self.msh)
        self.define_boundary()
        self.set_up_problem()
    
    def define_boundary(self):
        # Locate dofs on the top boundary
        self.fdim = self.msh.topology.dim - 1
        self.facets_top = mesh.locate_entities_boundary(self.msh, self.fdim, lambda x: np.isclose(x[1], 1.0))
        self.facets_bottom = mesh.locate_entities_boundary(self.msh, self.fdim, lambda x: np.isclose(x[1], 0.0))        
    
    def set_up_problem(self):
        self.f = self.MU[1].value * exp(-self.MU[2].value * ((self.x[0] - self.MU[3].value)**2 + 
                                                             (self.x[1] - self.MU[4].value)**2))
        dx = Measure("dx", self.msh)
        self.a = inner(self.sigma, self.tau) * dx + inner(self.u, div(self.tau)) * dx + inner(div(self.sigma), self.v) * dx
        self.L = -inner(self.f, self.v) * dx

    def f1(self, x):
        values = np.zeros((2, x.shape[1]))
        values[1, :] = np.sin(self.MU[0].value * x[0])
        return values
    
    def f2(self, x):
        values = np.zeros((2, x.shape[1]))
        values[1, :] = -np.sin(self.MU[0].value * x[0])
        return values
    
    def _inner_product_action(self, fun_j):
        def _(fun_i):
            return fun_i.vector.dot(fun_j.vector)
        return _
    
    def solve(self, mu_values):
        # Update MU values
        for i in range(len(mu_values)):
            self.MU[i].value = mu_values[i]

        # Get subspace of V
        V0 = self.V.sub(0)
        Q, _ = V0.collapse()
        dofs_top = fem.locate_dofs_topological((V0, Q), self.fdim, self.facets_top)

        f_h1 = fem.Function(Q)
        f_h1.interpolate(self.f1)
        bc_top = fem.dirichletbc(f_h1, dofs_top, V0)

        dofs_bottom = fem.locate_dofs_topological((V0, Q), self.fdim, self.facets_bottom)

        f_h2 = fem.Function(Q)
        f_h2.interpolate(self.f2)
        bc_bottom = fem.dirichletbc(f_h2, dofs_bottom, V0)

        bcs = [bc_top, bc_bottom]

        # Solve the linear problem
        self.set_up_problem()

        ### For mumps solver_type: https://petsc.org/main/manualpages/Mat/MATSOLVERMUMPS/
        problem = LinearProblem(self.a, self.L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                                        "pc_factor_mat_solver_type": "superlu"})
        
        try:
            w_h = problem.solve()
            # print("successfully solved")
            return w_h
        except PETSc.Error as e:
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

class PODReducedProblem:
    def __init__(self, problem):
        V, _ = problem.V.sub(0).collapse()
        Q, _ = problem.V.sub(1).collapse()
        self._basis_functions_sigma = rbnicsx.backends.FunctionsList(V)
        self._basis_functions_u = rbnicsx.backends.FunctionsList(Q)
        sigma, u = ufl.TrialFunction(V), ufl.TrialFunction(Q)
        v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
        self._inner_product_sigma = inner(sigma, v) * dx + \
            inner(div(sigma), div(v)) * dx
        self._inner_product_action_sigma = \
            rbnicsx.backends.bilinear_form_action(self._inner_product_sigma,
                                                  part="real")
        self._inner_product_u = inner(u, q) * dx
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

def write_file_u(problem, file_name, solution_u):
    print("Writing file:",file_name)
    with io.XDMFFile(problem.msh.comm, file_name, "w") as file:
        file.write_mesh(problem.msh)
        file.write_function(solution_u)

def write_file_sigma(problem, file_name, solution_sigma):
    print("Writing file:",file_name)
    Sigma_plot_element = element("Lagrange", problem.msh.basix_cell(), problem.k, shape=(problem.msh.geometry.dim,))
    Sigma_plot = fem.functionspace(problem.msh, Sigma_plot_element)
    sigma_h_plot = fem.Function(Sigma_plot)
    sigma_h_expr = fem.Expression(solution_sigma, Sigma_plot.element.interpolation_points())
    sigma_h_plot.interpolate(sigma_h_expr)
 
    with io.XDMFFile(problem.msh.comm, file_name, "w") as file:
        file.write_mesh(problem.msh)
        file.write_function(sigma_h_plot)

"""
### Test Solver
solver = MixedPoissonSolver()
mu_values_1 = [5.0, 10.0] 
mu_values_2 = [5.0, 11.0]
w_h_1 = solver.solve(mu_values_1)
w_h_2 = solver.solve(mu_values_2)
sigma_h_1, u_h_1 = w_h_1.split()
sigma_h_2, u_h_2 = w_h_2.split()
sigma_h_1 = sigma_h_1.collapse()
u_h_1 = u_h_1.collapse()
sigma_h_2 = sigma_h_2.collapse()
u_h_2 = u_h_2.collapse()
print(len(sigma_h_1.x.array))
print(len(sigma_h_2.x.array))
print(len(u_h_1.x.array))
print(len(u_h_2.x.array))

# If you are measuring error between ```u_h_1''' and ```u_h_2'''
error_u = fem.Function(solver.V.sub(1).collapse()[0])
error_u.x.array[:] = abs(u_h_1.x.array - u_h_2.x.array)

# If you are measuring error between ```sigma_h_1''' and ```sigma_h_2'''
error_sigma = fem.Function(solver.V.sub(0).collapse()[0])
error_sigma.x.array[:] = abs(sigma_h_1.x.array - sigma_h_2.x.array)

write_file_u(solver,"out_mixed_poisson/trial.xdmf", error_u)
write_file_sigma(solver,"out_mixed_poisson/sigma_trial.xdmf", error_sigma)
"""

def generate_training_set(sample_size = [3, 3, 3, 3, 3]):
    # Generate input parameter matrix for MU, depending on sample_size
    set_1 = np.linspace(5, 20, sample_size[0])
    set_2 = np.linspace(10, 20, sample_size[1])
    set_3 = np.linspace(10, 50, sample_size[2])
    set_4 = np.linspace(0.3, 0.7, sample_size[3])
    set_5 = np.linspace(0.3, 0.7, sample_size[4])
    training_set = np.array(list(itertools.product(set_1,set_2, set_3,
                                                        set_4, set_5)))
    return training_set

### Random parametrisation for POD?

def generate_training_set_2(sample_size = [10,10]):
    # Generate input parameter matrix for MU, depending on sample_size
    set_1 = np.linspace(2*np.pi, 2.5*np.pi, sample_size[0])
    set_2 = np.linspace(12, 13, sample_size[1])
    training_set = np.array(list(itertools.product(set_1,set_2)))
    return training_set

"""
def generate_training_set_2(sample_size = [10,10]):
    # Generate input parameter matrix for MU, depending on sample_size
    set_1 = np.random.rand(sample_size[0]) * (10./4. - 2) * np.pi + 2. * np.pi
    set_2 = np.random.rand(sample_size[1]) * (13. - 12.) + 12.
    training_set = np.array(list(itertools.product(set_1,set_2)))
    return training_set

"""

    
def create_training_snapshots(training_set, problem_parametric):
    print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
    print("")
    print("Set up snapshots matrix")
    V, _ = problem_parametric.V.sub(0).collapse()
    Q, _ = problem_parametric.V.sub(1).collapse()
    snapshots_matrix_sigma = rbnicsx.backends.FunctionsList(V)
    snapshots_matrix_u = rbnicsx.backends.FunctionsList(Q)

    for (mu_index, mu) in enumerate(training_set):
        print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))
        print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
        print("High fidelity solve for mu =", mu)
        w_h = problem_parametric.solve(mu)
        snapshot_sigma, snapshot_u = w_h.split()
        snapshot_sigma = snapshot_sigma.collapse()
        snapshot_u = snapshot_u.collapse()
        print("Update snapshots matrix")
        snapshots_matrix_sigma.append(snapshot_sigma)
        snapshots_matrix_u.append(snapshot_u)
    
    return snapshots_matrix_sigma, snapshots_matrix_u

problem_parametric = MixedPoissonSolver()
reduced_problem = PODReducedProblem(problem_parametric)
SAMPLE_SIZE = [10, 10]

# training_set = generate_training_set(SAMPLE_SIZE)
start_time = time.time()
training_set = generate_training_set_2(SAMPLE_SIZE)
snapshots_matrix_sigma, snapshots_matrix_u = create_training_snapshots(training_set, problem_parametric)
end_time = time.time()

print("Time for generating 100 samples in serial is:", end_time-start_time)
Nmax = 30
start_time_u = time.time()
print(rbnicsx.io.TextLine("Perform POD", fill="#"))
eigenvalues_u, modes_u, _ = rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_u,
                                    reduced_problem._inner_product_action_u,
                                    N=Nmax, tol=1e-6)

end_time_u = time.time()

start_time_sigma = time.time()
eigenvalues_sigma, modes_sigma, _ = rbnicsx.backends.\
    proper_orthogonal_decomposition(snapshots_matrix_sigma,
                                    reduced_problem._inner_product_action_sigma,
                                    N=Nmax, tol=1e-6)
end_time_sigma = time.time()


def plotting_eigenvalues(eigen_u, eigen_sigma, num=50, fontsize=14):
    
    top_eigen_u = eigen_u[:num]
    top_eigen_sigma = eigen_sigma[:num]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    index = range(1, num + 1)
    axes[0].plot(index, top_eigen_u, marker='o', linestyle='-')
    axes[0].set_title(f'First {num} Eigenvalues for $u$', fontsize=fontsize)
    axes[0].set_xlabel('Index', fontsize=fontsize)
    axes[0].set_ylabel('Eigenvalue', fontsize=fontsize)
    axes[0].tick_params(axis='both', which='major', labelsize=fontsize)
    
    axes[1].plot(index, top_eigen_sigma, marker='o', linestyle='-')
    axes[1].set_title(f'First {num} Eigenvalues for $\sigma$', fontsize=fontsize)
    axes[1].set_xlabel('Index', fontsize=fontsize)
    axes[1].set_ylabel('Eigenvalue', fontsize=fontsize)
    axes[1].tick_params(axis='both', which='major', labelsize=fontsize)
    
    # Set y-axis to log scale
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

reduced_problem._basis_functions_u.extend(modes_u)
reduced_problem._basis_functions_sigma.extend(modes_sigma)
print("First 20 eigenvalues for sigma:", eigenvalues_sigma[:20])
print("First 20 eigenvalues for u:", eigenvalues_u[:20])
print(f"Active number of modes for u and sigma: {len(modes_u)}, {len(modes_sigma)}") 
print(f"Elapsed time for u is {(end_time_u - start_time_u): 4f}")   
print(f"Elapsed time for sigma is {(end_time_sigma - start_time_sigma): 4f}")      
plotting_eigenvalues(eigenvalues_u, eigenvalues_sigma)

def calculate_error_u(problem, reduced_problem, original_solution, rb_solution):
    regenerated_solution = reduced_problem.reconstruct_solution_u(rb_solution)
    error = fem.Function(problem.V.sub(1).collapse()[0])
    error = regenerated_solution - original_solution
    norm_error =  reduced_problem.compute_norm_u(error) / reduced_problem.compute_norm_u(original_solution)
    return norm_error

def calculate_error_sigma(problem, reduced_problem, original_solution, rb_solution):
    regenerated_solution = reduced_problem.reconstruct_solution_sigma(rb_solution)
    error = fem.Function(problem.V.sub(0).collapse()[0])
    error = regenerated_solution - original_solution
    norm_error = reduced_problem.compute_norm_sigma(error)/reduced_problem.compute_norm_sigma(original_solution)
    return norm_error

def generate_ann_input_set(num_samples = 32):
    limits = np.array([[5, 20], [5, 20],
                       [10, 50], [0.2, 0.8], 
                       [0.2, 0.8]])
    sampling = LHS(xlimits=limits)
    x = sampling(num_samples)
    return x

def generate_ann_input_set_2(num_samples = 32):
    limits = np.array([[2.*np.pi, 10./4.*np.pi], [12., 13.]])
    sampling = LHS(xlimits=limits)
    x = sampling(num_samples)
    return x

def generate_ann_output_set(problem_parametric, reduced_problem, ann_training_set):
    output_set_sigma = np.empty([ann_training_set.shape[0], len(reduced_problem._basis_functions_sigma)])
    output_set_u = np.empty([ann_training_set.shape[0], len(reduced_problem._basis_functions_u)])
    print(f"Size of output set u and sigma are: {output_set_u.shape}, {output_set_sigma.shape}")
    rb_size_sigma = len(reduced_problem._basis_functions_sigma)
    rb_size_u = len(reduced_problem._basis_functions_u)
    print(f"Size of reduced basis u and sigma are: {rb_size_u}, {rb_size_sigma}")
    errors_sigma = np.zeros(ann_training_set.shape[0], dtype=np.float64)
    errors_u = np.zeros(ann_training_set.shape[0], dtype=np.float64)
    original_solutions = {}

    for i in range(ann_training_set.shape[0]):
        if i % 20 == 0:
            print(f"Parameter number {i+1} of {ann_training_set.shape[0]}: {ann_training_set[i,:]}")
        mu = ann_training_set[i]
        w_h = problem_parametric.solve(mu)
        solution_sigma, solution_u = w_h.split()
        solution_sigma = solution_sigma.collapse()
        solution_u = solution_u.collapse()
        rb_solution_sigma = reduced_problem.project_snapshot_sigma(solution_sigma, rb_size_sigma)
        rb_solution_u = reduced_problem.project_snapshot_u(solution_u, rb_size_u)
        output_set_sigma[i, :] = rb_solution_sigma.array  
        output_set_u[i, :] = rb_solution_u.array
        original_solutions[tuple(mu)] = [solution_u, solution_sigma]
        
        if i % 15 == 0:
            regenerated_solution_u = reduced_problem.reconstruct_solution_u(rb_solution_u)
            regenerated_solution_sigma = reduced_problem.reconstruct_solution_sigma(rb_solution_sigma)
            write_file_u(problem_parametric, f"test/FEMU{i}.xdmf", solution_u)
            write_file_u(problem_parametric, f"test/PODU{i}.xdmf", regenerated_solution_u)
            write_file_sigma(problem_parametric, f"test/FEMSIGMA{i}.xdmf", solution_sigma)
            write_file_sigma(problem_parametric, f"test/PODSIGMA{i}.xdmf", regenerated_solution_sigma)
        

        # u_error = reduced_problem.norm_error_u(solution_u, rb_solution_u)
        # sigma_error = reduced_problem.norm_error_sigma(solution_sigma, rb_solution_sigma)
        u_error = calculate_error_u(problem_parametric, reduced_problem, solution_u, rb_solution_u)
        sigma_error = calculate_error_sigma(problem_parametric, reduced_problem, solution_sigma, rb_solution_sigma)
        errors_u[i] = u_error
        errors_sigma[i] = sigma_error

    return output_set_u, output_set_sigma, errors_u, errors_sigma, original_solutions


INPUT_SHAPE = 2
NUM_ANN_SAMPLES = 32
ann_input_set = generate_ann_input_set_2(NUM_ANN_SAMPLES)
# ann_input_set = generate_ann_input_set(NUM_ANN_SAMPLES)

ann_output_set_u, ann_output_set_sigma, POD_errors_u, POD_errors_sigma, highfid_solutions = \
                            generate_ann_output_set(problem_parametric, reduced_problem, ann_input_set)
print(f"The mean error for U from POD on the inputs is {np.mean(POD_errors_u): 4f}, and the maximum error is {np.max(POD_errors_u): 4f}")
print(f"The mean error for SIGMA from POD on the inputs is {np.mean(POD_errors_sigma): 4f}, and the maximum error is {np.max(POD_errors_sigma): 4f}")
# Update the input and output ranges by looking for the max and min values
# Outputs range can be updated by directly look for the single max and min
# Inputs range is updated using specific methods
reduced_problem.output_range_u[0] = np.min(ann_output_set_u)
reduced_problem.output_range_u[1] = np.max(ann_output_set_u)
reduced_problem.output_range_sigma[0] = np.min(ann_output_set_sigma)
reduced_problem.output_range_sigma[1] = np.max(ann_output_set_sigma)
exit()
# reduced_problem.update_input_range(ann_input_set)

print(ann_output_set_u.shape)

customDataset_u = CustomDataset(reduced_problem, ann_input_set, ann_output_set_u,
                            input_scaling_range = reduced_problem.input_scaling_range,
                            output_scaling_range = reduced_problem.output_scaling_range_u,
                            input_range = reduced_problem.input_range,
                            output_range = reduced_problem.output_range_u,
                            verbose = False)

customDataset_sigma = CustomDataset(reduced_problem, ann_input_set, ann_output_set_sigma,
                                    input_scaling_range = reduced_problem.input_scaling_range,
                                    output_scaling_range = reduced_problem.output_scaling_range_sigma,
                                    input_range = reduced_problem.input_range,
                                    output_range = reduced_problem.output_range_sigma,
                                    verbose = False)

# Sigma and u should share the same scaled inputs
scaled_inputs_u = customDataset_u.input_transform(customDataset_u.input_set)
scaled_outputs_u = customDataset_u.output_transform(customDataset_u.output_set)
scaled_inputs_sigma = customDataset_sigma.input_transform(customDataset_sigma.input_set)
scaled_outputs_sigma = customDataset_sigma.output_transform(customDataset_sigma.output_set)

BATCH_SIZE = 16

train_dataloader_u, test_dataloader_u = create_Dataloader(scaled_inputs_u, scaled_outputs_u, 
                                                          train_batch_size=BATCH_SIZE, test_batch_size=1)
train_dataloader_sigma, test_dataloader_sigma = create_Dataloader(scaled_inputs_sigma, scaled_outputs_sigma, 
                                                                  train_batch_size=BATCH_SIZE, test_batch_size=1)

def plot_loss_over_epochs(train_results, name, fontsize=16, labelsize = 14):
    train_loss = train_results['train_loss']
    test_loss = train_results['test_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title(name, fontsize=fontsize)
    plt.xlabel('Epochs', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend(fontsize=labelsize)
    plt.show()

def plot_loss_over_epochs(train_results, name, fontsize=16, labelsize=14):
    train_loss = train_results['train_loss']
    test_loss = train_results['test_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Validation Loss')
    plt.title(name, fontsize=fontsize)
    plt.yscale('log')  # Set the y-axis to a logarithmic scale
    plt.xlabel('Epochs', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)
    plt.legend(fontsize=labelsize)
    plt.show()

# Visualise the shape of example batch in the dataloaders
for X, y in train_dataloader_sigma:
    print(f"Shape of SIGMA training set input: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"Shape of SIGMA training set output: {y.shape}")
    print(f"y dtype: {y.dtype}")
    break

for X, y in train_dataloader_u:
    print(f"Shape of U training set input: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"Shape of U training set output: {y.shape}")
    print(f"y dtype: {y.dtype}")
    break

### TRAINING FOR MODEL U
neurons_u = [25,25]
model_u = create_model(input_shape = INPUT_SHAPE, output_shape = len(reduced_problem._basis_functions_u), 
                           hidden_layers_neurons = neurons_u, activation='tanh')

lr = 5.e-4
# loss_object = tf.keras.losses.MeanAbsoluteError()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

NUM_EPOCH = 1000

result_dict_u = train(model_u,
                    train_dataloader=train_dataloader_u,
                    test_dataloader=test_dataloader_u,
                    optimizer=optimizer,
                    loss_fn=loss_object,
                    epochs=NUM_EPOCH,
                    device="cpu",
                    early_stopping = 3)

### TRAINING FOR MODEL SIGMA
neurons_sigma = [15,15]
model_sigma = create_model(input_shape = INPUT_SHAPE, output_shape = len(reduced_problem._basis_functions_sigma), 
                           hidden_layers_neurons = neurons_sigma, activation='tanh')
# loss_object = tf.keras.losses.MeanAbsoluteError()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

NUM_EPOCH = 1000
result_dict_sigma = train(model_sigma,
                        train_dataloader=train_dataloader_sigma,
                        test_dataloader=test_dataloader_sigma,
                        optimizer=optimizer,
                        loss_fn=loss_object,
                        epochs=NUM_EPOCH,
                        device="cpu",
                        early_stopping = 3)

name_u = f"Losses for $u$, lr={lr}, ANN Sturcture: {neurons_u}"
plot_loss_over_epochs(result_dict_u, name_u)

name_sigma = f"Losses for $\sigma$, lr={lr}, ANN Sturcture: {neurons_sigma}"
plot_loss_over_epochs(result_dict_sigma, name_sigma)

# error_analysis_mu_set = generate_training_set(sample_size = [2,2,2,2,2])

# loaded_model = tf.keras.models.load_model('sigma_model')

print("\n")
print("Generating error analysis (only input/parameters) dataset")
print("\n")
# error_analysis_set = generate_ann_input_set(100)
error_analysis_set = generate_ann_input_set_2(50)
error_numpy_u = np.zeros(error_analysis_set.shape[0])
error_numpy_sigma = np.zeros(error_analysis_set.shape[0])
print(error_analysis_set.shape[0])
ann_solutions_u = [None]*error_analysis_set.shape[0]
ann_solutions_sigma = [None]*error_analysis_set.shape[0]
offline_time = np.zeros(error_analysis_set.shape[0])
online_time = np.zeros(error_analysis_set.shape[0])
for i in range(error_analysis_set.shape[0]):
    print(f"Error analysis parameter number {i+1} of ")
    print(f"{error_analysis_set.shape[0]}: {error_analysis_set[i,:]}")
    fem_start = time.time()
    w_h = problem_parametric.solve(error_analysis_set[i, :])
    solution_sigma, solution_u = w_h.split()
    fem_solution_sigma = solution_sigma.collapse()
    fem_solution_u = solution_u.collapse()
    fem_end = time.time()
    fem_elapsed_time = fem_end - fem_start
    offline_time[i] = fem_elapsed_time
    error_numpy_u[i], ann_solutions_u[i], online_time_u = error_analysis(reduced_problem, problem_parametric,
                                                        error_analysis_set[i, :], model_u,
                                                        len(reduced_problem._basis_functions_u), 
                                                        online_nn, fem_solution_u,
                                                        norm_error=reduced_problem.norm_error_u,
                                                        reconstruct_solution=reduced_problem.reconstruct_solution_u,
                                                        input_scaling_range=reduced_problem.input_scaling_range,
                                                        output_scaling_range=reduced_problem.output_scaling_range_u,
                                                        input_range=reduced_problem.input_range,
                                                        output_range=reduced_problem.output_range_u)
    
    error_numpy_sigma[i], ann_solutions_sigma[i],  online_time_sigma = error_analysis(reduced_problem, problem_parametric, 
                                                                error_analysis_set[i, :], model_sigma,
                                                                len(reduced_problem._basis_functions_sigma), 
                                                                online_nn, fem_solution_sigma,
                                                                norm_error=reduced_problem.norm_error_sigma,
                                                                reconstruct_solution=reduced_problem.reconstruct_solution_sigma, 
                                                                input_scaling_range=reduced_problem.input_scaling_range, 
                                                                output_scaling_range=reduced_problem.output_scaling_range_sigma, 
                                                                input_range=reduced_problem.input_range,
                                                                output_range=reduced_problem.output_range_sigma)
    print(f"Error for U: {error_numpy_u[i]}")
    print(f"Error for SIGMA: {error_numpy_sigma[i]}")
    total_online_time = online_time_u + online_time_sigma
    online_time[i] = total_online_time

    ### Plotting FEM, POD_reconsturcted and ANN solutions for comparison
    if i % 20 == 0:
        rb_size_sigma = len(reduced_problem._basis_functions_sigma)
        rb_size_u = len(reduced_problem._basis_functions_u)
        rb_solution_sigma = reduced_problem.project_snapshot_sigma(fem_solution_sigma, rb_size_sigma)
        rb_solution_u = reduced_problem.project_snapshot_u(fem_solution_u, rb_size_u)
        regenerated_solution_u = reduced_problem.reconstruct_solution_u(rb_solution_u)
        regenerated_solution_sigma = reduced_problem.reconstruct_solution_sigma(rb_solution_sigma)
        error_u = fem.Function(problem_parametric.V.sub(1).collapse()[0])
        error_u.x.array[:] = regenerated_solution_u.x.array - fem_solution_u.x.array
        error_sigma = fem.Function(problem_parametric.V.sub(0).collapse()[0])
        error_sigma.x.array[:] = regenerated_solution_sigma.x.array - fem_solution_sigma.x.array
        write_file_u(problem_parametric, f"Solutions_Comparison_Final/FEM_U{i}.xdmf", fem_solution_u)
        write_file_u(problem_parametric, f"Solutions_Comparison_Final/POD_U{i}.xdmf", regenerated_solution_u)
        write_file_u(problem_parametric, f"Solutions_Comparison_Final/ANN_U{i}_error={error_numpy_u[i]}.xdmf", ann_solutions_u[i])
        write_file_u(problem_parametric, f"Solutions_Comparison_Final/Error_U{i}_error={error_numpy_u[i]}.xdmf", error_u)
        write_file_sigma(problem_parametric, f"Solutions_Comparison_Final/FEM_SIGMA{i}.xdmf", fem_solution_sigma)
        write_file_sigma(problem_parametric, f"Solutions_Comparison_Final/POD_SIGMA{i}.xdmf", regenerated_solution_sigma)
        write_file_sigma(problem_parametric, f"Solutions_Comparison_Final/ANN_SIGMA{i}_error={error_numpy_sigma[i]}.xdmf", ann_solutions_sigma[i])
        write_file_sigma(problem_parametric, f"Solutions_Comparison_Final/Error_SIGMA{i}_error={error_numpy_sigma[i]}.xdmf", error_sigma)

print(f"Final results for a tanh activation function with {NUM_ANN_SAMPLES} training and validation samples:")
print(f"Learing rate = {lr}, ANN structure sigma:{neurons_sigma}, u: {neurons_u}")
print(f"Mean error for U is: {np.mean(error_numpy_u):4f}; Mean Error for SIGMA is {np.mean(error_numpy_sigma):4f}")
print(f"Maximum error for U and SIGMA are: {np.max(error_numpy_u):4f}, {np.max(error_numpy_sigma):4f}")
print(f"Mean offline and online time is: {np.max(offline_time):4f}, {np.max(online_time):4f}")