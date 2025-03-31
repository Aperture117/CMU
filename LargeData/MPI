import numpy as np
from mpi4py import MPI
import math

ROWS , COLUMNS = 1000 , 1000

#MPI Specific Additions
NPES = 4
ROWS = int(ROWS/NPES)
NPES = 4
DOWN = 100
UP = 101

MAX_TEMP_ERROR = 0.01
temperature = np.empty(( ROWS+2 , COLUMNS+2 ))
temperature_last = np.empty(( ROWS+2 , COLUMNS+2 ))
def initialize_temperature(temp):
    temp[:,:] = 0
    #Set right side boundary condition
    start = (my_PE_num) * (3.14159/2) /NPES
    step = (3.14159/2) / (ROWS*NPES)
    
    for i in range(ROWS+1):
        temp[ i , COLUMNS+1 ] = 100 * math.sin( start + i * step)
    #Set bottom boundary condition on bottom PE
    if my_PE_num == (NPES-1):
        for i in range(COLUMNS+1):
            temp[ ROWS+1 , i ] = 100 * math.sin( ( (3.14159/2) /COLUMNS ) * i )
            
def output(data):
    for rank in range(NPES):
        comm.barrier()
        if my_PE_num == rank:
            if my_PE_num == 0: data_out = data[:-1,:]
            if my_PE_num == 1: data_out = data[1:-1,:]
            if my_PE_num == 2: data_out = data[1:-1,:]
            if my_PE_num == 3: data_out = data[1:,:]
            shared_file = open("plate.out","ab")
            data_out.tofile(shared_file)
            shared_file.close()
            
comm = MPI.COMM_WORLD
my_PE_num = comm.Get_rank()

initialize_temperature(temperature_last)

if my_PE_num == 0:
#hardwire this for slurm jobs
    max_iterations = int (input("Maximum iterations: "))
else:
    max_iterations = None
max_iterations = comm.bcast(max_iterations, root=0)

dt = 100
iteration = 1

while ( dt > MAX_TEMP_ERROR ) and ( iteration < max_iterations ):
    for i in range( 1 , ROWS+1 ):
        for j in range( 1 , COLUMNS+1 ):
            temperature[ i , j ] = 0.25 * ( temperature_last[i+1,j] + temperature_last[i-1,j] +
            temperature_last[i,j+1] + temperature_last[i,j-1] )
# send bottom real row down
    if my_PE_num != NPES-1:
        comm.Send( temperature[ ROWS , 1:COLUMNS+1 ], dest=my_PE_num+1, tag=DOWN)
    # receive the bottom row from above into our top ghost row
    if my_PE_num != 0:
        comm.Recv( temperature_last[ 0 , 1:COLUMNS+1 ], source=my_PE_num-1, tag=DOWN)
    # send top real row up
    if my_PE_num != 0:
        comm.Send( temperature[ 1 , 1:COLUMNS+1 ], dest=my_PE_num-1, tag=UP)
    # receive the top row from below into our bottom ghost row
    if my_PE_num != NPES-1:
        comm.Recv( temperature_last[ ROWS+1 , 1:COLUMNS+1 ], source=my_PE_num+1, tag=UP)

    dt = 0

    for i in range( 1 , ROWS+1 ):
        for j in range( 1 , COLUMNS+1 ):
            dt = max( dt, temperature[i,j] - temperature_last[i,j])
            temperature_last[ i , j ] = temperature [ i , j ]

    dt = comm.allreduce( dt, op=MPI.MAX )

    if my_PE_num==0:print(iteration, flush=True)
    iteration += 1
    
output(temperature_last)
