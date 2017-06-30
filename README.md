The size of the matrix is determined by the variable block. This is a fixed value in the code
(# define block 180)

Compiled using mpicc -fopenmp -0 file file.c. The number
of threads was specified using export OMP NUM THREADS=x
