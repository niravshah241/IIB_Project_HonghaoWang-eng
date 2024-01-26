from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import LinearProblem
from ufl import Measure, SpatialCoordinate, TestFunctions, TrialFunctions, div, exp, inner
import rbnicsx
import rbnicsx.backends
import itertools

class MixedPoissonSolver:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.msh = mesh.create_unit_square(self.comm, 32, 32, mesh.CellType.quadrilateral)
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
        print(values)
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

        ### TOASK: the different code here in LinearProblem
        problem = LinearProblem(self.a, self.L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                                        "pc_factor_mat_solver_type": "superlu"})
        
        try:
            w_h = problem.solve()
            print("successfully solved")
            return w_h
        except PETSc.Error as e:
            if e.ierr == 92:
                print("The required PETSc solver/preconditioner is not available. Exiting.")
                print(e)
                exit(0)
            else:
                raise e

solver = MixedPoissonSolver()
"""
### TEST generate_training_set
sample_size = [3,3,3,3,3]
x = solver.generate_training_set(sample_size)
print(x)
"""

mu_values = [5.0, 10.0, 1.0/0.02, 0.5, 0.5] 
w_h = solver.solve(mu_values)

sigma_h, u_h = w_h.split()

with io.XDMFFile(solver.msh.comm, "out_mixed_poisson/trial.xdmf", "w") as file:
    file.write_mesh(solver.msh)
    file.write_function(u_h)

Sigma_plot_element = element("Lagrange", solver.msh.basix_cell(), solver.k, shape=(solver.msh.geometry.dim,))
Sigma_plot = fem.FunctionSpace(solver.msh, Sigma_plot_element)
sigma_h_plot = fem.Function(Sigma_plot)
sigma_h_expr = fem.Expression(sigma_h, Sigma_plot.element.interpolation_points())
sigma_h_plot.interpolate(sigma_h_expr)
 
with io.XDMFFile(solver.msh.comm, "out_mixed_poisson/sigma_trial.xdmf", "w") as file:
    file.write_mesh(solver.msh)
    file.write_function(sigma_h_plot)

def generate_training_set(sample_size):
    # Generate input parameter matrix for MU, depending on sample_size
    set_1 = np.linspace(5, 20, sample_size[0])
    set_2 = np.linspace(10, 20, sample_size[1])
    set_3 = np.linspace(10, 50, sample_size[2])
    set_4 = np.linspace(0.3, 0.7, sample_size[3])
    set_5 = np.linspace(0.3, 0.7, sample_size[4])
    training_set = np.array(list(itertools.product(set_1,set_2, set_3,
                                                        set_4, set_5)))
    return training_set
    
def create_training_set(solver, sample_size):
    # Generate the training set matrix before performing POD
    # training_set = rbnicsx.io.on_rank_zero(mesh.comm, generate_training_set)
    training_set = generate_training_set(sample_size)
    # print(rbnicsx.io.TextBox("POD offline phase begins", fill="="))
    print("")
    print("Set up snapshots matrix")
    V, _ = solver.V.sub(0).collapse()
    Q, _ = solver.V.sub(1).collapse()
    snapshots_matrix_sigma = rbnicsx.backends.FunctionsList(V)
    snapshots_matrix_u = rbnicsx.backends.FunctionsList(Q)

    for (mu_index, mu) in enumerate(training_set):
        # print(rbnicsx.io.TextLine(str(mu_index+1), fill="#"))

        print("Parameter number ", (mu_index+1), "of", training_set.shape[0])
        print("High fidelity solve for mu =", mu)
        w_h = solver.solve(mu)
        snapshot_sigma, snapshot_u = w_h.split()
        print("Update snapshots matrix")
        snapshots_matrix_sigma.append(snapshot_sigma)
        snapshots_matrix_u.append(snapshot_u)
    
    return snapshot_sigma, snapshot_u


sample_size = [3,3,3,3,3]
a, b = create_training_set(solver, sample_size)
print(a.shape, b.shape)

