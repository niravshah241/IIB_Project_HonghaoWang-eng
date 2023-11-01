from mpi4py import MPI
import numpy as np

rows = 100
cols = 100
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

A = np.random.rand(rows, cols)
# print(f"Current rank: {rank}, Local data: {A}")
comm.Barrier()

# Use Allreduce to compute the sum of A across all processes
start_time = MPI.Wtime()
result = comm.allreduce(A, op=MPI.SUM) #replace MPI.PROD for product of matrices

end_time = MPI.Wtime()
elapsed_time = end_time - start_time

print(f"Elapsed time for rank {rank} is {elapsed_time: 4f} seconds")

"""
comm.Barrier()

if rank == 0:
    print(f"global sum = {result}")
"""