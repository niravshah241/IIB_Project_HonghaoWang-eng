from mpi4py import MPI
from petsc4py import PETSc

import ufl
import basix
import dolfinx
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, \
    apply_lifting, set_bc

import numpy as np

L = 20.0
domain = dolfinx.mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, 1, 1]], [20, 5, 5], dolfinx.mesh.CellType.tetrahedron)
# V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2, (domain.geometry.dim, )))
V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 2, domain.geometry.dim))

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh_data/mesh.xdmf", "w") as mesh_file_xdmf:
    mesh_file_xdmf.write_mesh(domain)
