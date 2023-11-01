from mpi4py import MPI
import numpy as np

rows = 12
cols = 12
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    A = np.random.rand(rows, cols)
else:
    A = None

print("Matrix A:", A)
# Synchronize all ranks
comm.Barrier()
A = comm.bcast(A, root=0)
local_indices = np.arange(comm.rank, A.shape[0], comm.size)
data = A[local_indices, :]

"""
# Take rows whose row numbers are in multiples of 3
if rank == 0:
    data = A[0: :3]

# Take rows whose row numbers modulo 3 is zero
elif rank == 1:
    data = A[1: :3]

elif rank == 2:
    data = A[2: :3]
"""

print(f"Rank {rank} reads data {data}")