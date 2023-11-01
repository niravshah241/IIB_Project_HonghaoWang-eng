# # Poisson equation
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner

from dolfinx import fem, io, mesh, plot

# -

# We create a rectangular {py:class}`Mesh <dolfinx.mesh.Mesh>` using
# {py:func}`create_rectangle <dolfinx.mesh.create_rectangle>`, and
# create a finite element {py:class}`FunctionSpace
# <dolfinx.fem.FunctionSpace>` $V$ on the mesh.

# +
msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (2.0, 1.0)), n=(32, 16),
                            cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 1))
# -

# The second argument to {py:class}`FunctionSpace
# <dolfinx.fem.FunctionSpace>` is a tuple `(family, degree)`, where
# `family` is the finite element family, and `degree` specifies the
# polynomial degree. In this case `V` is a space of continuous Lagrange
# finite elements of degree 1.
#
# To apply the Dirichlet boundary conditions, we find the mesh facets
# (entities of topological co-dimension 1) that lie on the boundary
# $\Gamma_D$ using {py:func}`locate_entities_boundary
# <dolfinx.mesh.locate_entities_boundary>`. The function is provided
# with a 'marker' function that returns `True` for points `x` on the
# boundary and `False` otherwise.

facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 2.0)))

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`:

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>` class that
# represents the boundary condition:

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

# Next, the variational problem is defined:

# +
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds
# -

# A {py:class}`LinearProblem <dolfinx.fem.LinearProblem>` object is
# created that brings together the variational problem, the Dirichlet
# boundary condition, and which specifies the linear solver. In this
# case an LU solver us sued. The {py:func}`solve
# <dolfinx.fem.LinearProblem.solve>` computes the solution.

# +
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
# -

# The solution can be written to a {py:class}`XDMFFile
# <dolfinx.io.XDMFFile>` file visualization with ParaView or VisIt:

# +
with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(uh)
# -

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
try:
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
# -
