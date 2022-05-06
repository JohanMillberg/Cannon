#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
    This script uses Cannon's algorithm to multiply two matrices given in an input file.
*/

int is_compatible(int matrix_size, int size);
int read_input(char *filename, double ***A, double ***B);
void matrix_multiply(double *A, double *B, int N, double **C);
void allocate_matrix(double ***A, int matrix_size);
int write_output(char *file_name, double **output, int matrix_size);

int main(int argc, char *argv[])
{
    char *input_name = argv[1];
    char *output_name = argv[2];
    int matrix_size;
    int rank, size;
    int sqrt_of_size;
    double **A;
    double **B;
    double **C;
    double *A_local_send;
    double *A_local_recv;
    double *B_local_send;
    double *B_local_recv;
    double *C_local;
    MPI_Comm comm2D;
    MPI_Status status;
    MPI_Request send_request_A, recv_request_A, send_request_B, recv_request_B;

    int dim[2];
    int period[2];
    int north, east, south, west;
    int coords[2];

    int broadcast_data[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Handle input of data
    if (rank == 0)
    {
        if (0 > (matrix_size = read_input(input_name, &A, &B)))
            return 2;

        allocate_matrix(&C, matrix_size);
        broadcast_data[0] = matrix_size;

        // Check if sqrt of size is integer, and if matrix size is divisible by the sqrt.
        if (!(is_compatible(matrix_size, size))) {
            return 2;
        }

        broadcast_data[1] = (int)sqrt(size);
    }

    MPI_Bcast(broadcast_data, 2, MPI_INT, 0, MPI_COMM_WORLD);

    matrix_size = broadcast_data[0];
    sqrt_of_size = broadcast_data[1];

    dim[0] = sqrt_of_size;
    dim[1] = sqrt_of_size;
    period[0] = 1;
    period[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, 1, &comm2D);

    int block_size = (int)matrix_size / sqrt_of_size;

    A_local_send = malloc(sizeof(double)*block_size*block_size);
    B_local_send = malloc(sizeof(double)*block_size*block_size);
    A_local_recv = malloc(sizeof(double)*block_size*block_size);
    B_local_recv = malloc(sizeof(double)*block_size*block_size);
    C_local = malloc(sizeof(double)*block_size*block_size);

    // Initialize the C matrix
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            C_local[i*block_size + j] = 0;
        }
    }

    int array_size[2] = {matrix_size, matrix_size};
    int local_size[2] = {block_size, block_size};
    int start_indices[2] = {0, 0};

    // Create the subarray datatype
    MPI_Datatype type, subarray;
    MPI_Type_create_subarray(2, array_size, local_size, start_indices, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_create_resized(type, 0, block_size * sizeof(double), &subarray);
    MPI_Type_commit(&subarray);

    int *send_count = malloc(sizeof(int) * size);
    int *displacement_count = malloc(sizeof(int) * size);

    double *A_pointer = NULL;
    double *B_pointer = NULL;
    double *C_pointer = NULL;

    /*
        Create the arrays that contain the amount of subarrays to send to each process
        as well as the displacement applied to the message sent to each process.
    */

    if (rank == 0)
    {
        int disp_count = 0;
        for (int i = 0; i < size; i++)
        {
            send_count[i] = 1;
        }

        // Set displacement of each process
        for (int i = 0; i < sqrt_of_size; i++)
        {
            for (int j = 0; j < sqrt_of_size; j++)
            {
                displacement_count[i*sqrt_of_size + j] = disp_count;
                disp_count++;
            }
            disp_count += (block_size - 1)*sqrt_of_size;
        }
        A_pointer = &(A[0][0]);
        B_pointer = &(B[0][0]);
        C_pointer = &(C[0][0]);
    }

    // Scatter the subarrays
    MPI_Scatterv(A_pointer, send_count, displacement_count, subarray, A_local_send,
                 matrix_size * matrix_size / (size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(B_pointer, send_count, displacement_count, subarray, B_local_send,
                 matrix_size * matrix_size / (size), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Initial shift
    MPI_Cart_coords(comm2D, rank, 2, coords);
    MPI_Cart_shift(comm2D, 1, coords[0], &west, &east);
    MPI_Cart_shift(comm2D, 0, coords[1], &north, &south);

    MPI_Isend(A_local_send, block_size*block_size, MPI_DOUBLE, west, 0, comm2D, &send_request_A);
    MPI_Irecv(A_local_recv, block_size*block_size, MPI_DOUBLE, east, 0, comm2D, &recv_request_A);

    MPI_Isend(B_local_send, block_size*block_size, MPI_DOUBLE, north, 1, comm2D, &send_request_B);
    MPI_Irecv(B_local_recv, block_size*block_size, MPI_DOUBLE, south, 1, comm2D, &recv_request_B);

    MPI_Wait(&send_request_A, &status);
    MPI_Wait(&recv_request_A, &status);
    MPI_Wait(&send_request_B, &status);
    MPI_Wait(&recv_request_B, &status);

    MPI_Cart_shift(comm2D, 1, 1, &west, &east);
    MPI_Cart_shift(comm2D, 0, 1, &north, &south);

    double time_start = MPI_Wtime();

    // Execute the algorithm
    matrix_multiply(A_local_recv, B_local_recv, block_size, &C_local);

    for (int iter = 0; iter < sqrt_of_size-1; iter++) {
        double* tmp_A = A_local_recv;
        double* tmp_B = B_local_recv;
        A_local_recv = A_local_send;
        B_local_recv = B_local_send;
        A_local_send = tmp_A;
        B_local_send = tmp_B;

        // Shift A one step left, and B one step up
        MPI_Isend(A_local_send, block_size*block_size, MPI_DOUBLE, west, 0, comm2D, &send_request_A);
        MPI_Irecv(A_local_recv, block_size*block_size, MPI_DOUBLE, east, 0, comm2D, &recv_request_A);

        MPI_Isend(B_local_send, block_size*block_size, MPI_DOUBLE, north, 1, comm2D, &send_request_B);
        MPI_Irecv(B_local_recv, block_size*block_size, MPI_DOUBLE, south, 1, comm2D, &recv_request_B);

        MPI_Wait(&recv_request_A, &status);
        MPI_Wait(&recv_request_B, &status);

        matrix_multiply(A_local_recv, B_local_recv, block_size, &C_local);

        MPI_Wait(&send_request_A, &status);
        MPI_Wait(&send_request_B, &status);
    }
	double max_time;
	double execution_time = MPI_Wtime() - time_start;

	// Find the largest execution time
	MPI_Reduce(&execution_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm2D);

    // Gather the result
    MPI_Gatherv(C_local, matrix_size*matrix_size / size, MPI_DOUBLE, C_pointer,
        send_count, displacement_count, subarray, 0, MPI_COMM_WORLD);

    // Handle output and write the computation time
    if (rank == 0) {
        printf("%lf\n", max_time);
        write_output(output_name, C, matrix_size);
        free(A);
        free(B);
        free(C);

    }
    free(A_local_recv);
    free(A_local_send);
    free(B_local_recv);
    free(B_local_send);
    free(C_local);
    free(send_count); free(displacement_count);

    MPI_Comm_free(&comm2D);
    MPI_Finalize();
}

// Checks if the square root of the given amount of processes is an integer, as well as if the matrix size is divisible by this root.
int is_compatible(int matrix_size, int size) {
    double sqrt_of_size = sqrt(size);
    if (sqrt_of_size - floor(sqrt_of_size) != 0) {
        printf("Square root of the amount of processes must be an integer.\n");
        return 0;
    }
    
    if ((matrix_size % (int)sqrt_of_size) != 0) {
        printf("Matrix size must be divisible by the square root of the amount of processes.\n");
        return 0;
    }

    return 1;
}

// Uses ikj-multiplication to multiply the matrices.
void matrix_multiply(double *A, double *B, int N, double **C) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double x = A[i*N+k];
            for (int j = 0; j < N; j++) {
                (*C)[i*N+j] += x * B[k*N+j];
            }
        }
    }
}

// Allocates memory for the given array such that the elements in the two dimensional array is stored contiguously.
void allocate_matrix(double ***A, int matrix_size)
{
    double *p = (double *)malloc(sizeof(double *) * matrix_size * matrix_size);
    *A = malloc(sizeof(double *) * matrix_size);

    for (int i = 0; i < matrix_size; i++)
    {
        (*A)[i] = &(p[i * matrix_size]);
    }
}

// Function for handling input. Inspired by the read_input function given in A1.
int read_input(char *filename, double ***A, double ***B)
{
    FILE *file;
    if (NULL == (file = fopen(filename, "r")))
    {
        printf("Error opening file.\n");
        return -1;
    }
    int matrix_size;
    if (EOF == fscanf(file, "%d", &matrix_size))
    {
        perror("Couldn't read matrix size from input file");
        return -1;
    }
    allocate_matrix(A, matrix_size);
    allocate_matrix(B, matrix_size);

    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            if (EOF == fscanf(file, "%lf", &((*A)[i][j])))
            {
                perror("Couldn't read elements from input file");
                return -1;
            }
        }
    }
    for (int i = 0; i < matrix_size; i++)
    {
        for (int j = 0; j < matrix_size; j++)
        {
            if (EOF == fscanf(file, "%lf", &((*B)[i][j])))
            {
                perror("Couldn't read elements from input file");
                return -1;
            }
        }
    }
    if (0 != fclose(file))
    {
        perror("Warning: couldn't close input file");
    }
    return matrix_size;
}

// Function for handling output. Inspired by the read_input function given in A1.
int write_output(char *file_name, double **output, int matrix_size) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            if (0 > fprintf(file, "%.6f ", output[i][j])) {
                perror("Couldn't write to output file");
            }
        }
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;
}