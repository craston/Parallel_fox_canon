#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

#define block 180
/* ----------------------------------------------------------------
Matrix multiplication
--------------------------------------------------------------------*/
void matmul(int size, int A[size][size], int B[size][size], int C[size][size]){
    register unsigned int i, j, k ;
    register int r ;
    //#pragma omp parallel for schedule(dynamic) private(i,j, k, r)
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
            r = C[i][j];
            for(k = 0; k < size; k++){
                    r += A[i][k]*B[k][j];
            }
            C[i][j] = r;
        }
    }
}
/* ----------------------------------------------------------------
Matrix printing
--------------------------------------------------------------------*/
void print_matrix(int size, int A[size][size]){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
                printf("%d ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void assign_matrix(int size, int A[size][size], int B[size][size]){
    register unsigned int i,j;
    for(i = 0; i < size; i++){
        for(j = 0; j < size; j++){
                A[i][j] =  B[i][j];
        }
    }
}

int main(int argc, char *argv[])
{
     /* Creating the a grid of pxp processes */
    int my_rank, my_grid_rank, size, coord[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    char processor_name[MPI_MAX_PROCESSOR_NAME];


    MPI_Comm grid_comm;
    int dim[2], period[2], reorder;
    dim[0] = 0; dim[1] = 0;
    int ndims = 2;
    MPI_Dims_create(size, 2, dim);											//Calculate dimensions of grid

  	//Checking is grid is square
    if (dim[0] != dim[1])
    {
        printf("Number of processes should be a square.\n");
        MPI_Finalize();
        return 1;
    }
    period[0] = 1 ; period[1] = 1;
    reorder=1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &grid_comm);	//Create grid
    MPI_Comm_rank(grid_comm, &my_grid_rank);								//Rank of process in grid
    MPI_Cart_coords(grid_comm, my_grid_rank, 2, coord);						//Co-ord of procss within grid
    
    /* Creating sub grid for rows*/
    int free_coords[2], row_rank;
    MPI_Comm row_comm;														//Row communicator
    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid_comm, free_coords, &row_comm);					
    MPI_Comm_rank(row_comm, &row_rank);

     /* Creating sub grid for cols*/
    int col_rank;
    MPI_Comm col_comm;
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid_comm, free_coords, &col_comm);
    MPI_Comm_rank(col_comm, &col_rank);

    /* Creating blocks locally in each process*/
    int A[block][block], B[block][block], C[block][block], Diag[block][block];
    double starttime, endtime;

    MPI_Datatype newtype;
    int my_sizes[2] = {dim[0]*block, dim[1]*block}; 						// Global matrix which the root will distribute
    int sub_sizes[2] = {block, block};   									// Submatrix size
    int start[2] = {0,0};													// Submatrix will start at (0,0)
    MPI_Type_create_subarray(2, my_sizes, sub_sizes, start, 1, MPI_INT, &newtype);	//Creating a new datatype
    MPI_Type_commit(&newtype);
    
    //Root Process initializing and distributing blocks of matrix to all processes of the grid
    if(my_rank == 0){
        int mainA[dim[0]*block][dim[1]*block];
		
		//Initializaing the matrix
        for(int i = 0; i < dim[0]*block; i++){
                for(int j = 0; j < dim[1]*block; j++){
                        mainA[i][j] = i;
                }
        }
        
        starttime = MPI_Wtime();								//Starting the timer
        //Sending the blocks to all the processes
        for(int i = 0; i < dim[0]; i++){                       
            for(int j = 0; j < dim[1] ; j++){
                    MPI_Send(&(mainA[i*block][j*block]), 1, newtype, dim[1]*i + j , 0, grid_comm);
            }
    	}
    }

    MPI_Recv(&(A[0][0]), block*block, MPI_INT, 0, 0, grid_comm, &status);		// 	Each Process receives the submatrix
//=======================================================================================================================
// BEGIN FOX ALGORITHM
//======================================================================================================================
    
    //Stage 0 Initializing matrix B and C
    
    for(int i = 0; i < block; i++){
        for(int j = 0; j < block; j++){
            B[i][j] = 1;
            C[i][j] = 0;
        }
    }
    //Finding the diagonal processes and storing the diagonal blocks
    if (coord[0] == coord[1]){
        assign_matrix(block, Diag, A);
    }
    
    //Broadcasting the diagonal A block to all the process in the row
    for(int i =0 ; i < dim[0]; i++){
        if(coord[0] == i)
        {
            MPI_Bcast(&Diag, block*block, MPI_INT, i, row_comm);
        }
    }
    //calculating C for stage 0 

   matmul(block, Diag, B, C);

   //====================================================================================================================
   //   STAGE 1.2, 3
   //=====================================================================================================================
   //Stage 1 : move B up 
   int stage = 1;
   while(stage <= dim[0]-1){
                
        MPI_Sendrecv_replace(&B, block*block, MPI_INT, (col_rank - 1 + dim[0])%dim[0], 0, (col_rank + 1 + dim[0])%dim[0], 0, col_comm, &status);

        //Broadcast diagonal of A
        for(int i = 0 ; i < dim[0]; i++)
        {
            if (coord[0]== i && coord[1] == (stage + i )%dim[1]){
                    assign_matrix(block, Diag, A);
            }
        }

        for(int i = 0; i < dim[0]; i++){
            if(coord[0] == i)
            {
                    MPI_Bcast(&Diag, block*block, MPI_INT, (i + stage)%dim[0], row_comm);
            }
        }
        //calculating C for stage 1 

        matmul(block, Diag, B, C);
        stage++ ;
    }
//========================================================================================================================
//Gathering matrix C
//========================================================================================================================
    MPI_Send(&(C[0][0]), block*block, MPI_INT, 0 , 0, grid_comm);  //Each process sending the C block to the root process.

    if(my_rank == 0){												//Root process gathering the final matrix
        int mainA[dim[0]*block][dim[1]*block];

        for(int i = 0; i < dim[0] ; i++){                       	//Receiving the blocks from all the processes
            for(int j = 0; j < dim[1]; j++){
                MPI_Recv(&(mainA[i*block][block*j]), 1, newtype, dim[1]*i + j , 0, grid_comm, &status);
            }
        }
        endtime = MPI_Wtime();
        printf("Total time = %f \n", endtime - starttime);
        print_matrix(dim[0]*block, mainA);							//Display the final matrix
    }
    MPI_Finalize();
}
                                          

