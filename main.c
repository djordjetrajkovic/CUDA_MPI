#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>

void launch_multiply(const float* a, float* b, int n);

int main(int argc, char** argv) {
    int rank, nprocs;
    int n = 1000000;
    int chunk;
    float *A, *B;
    float *pA, *pB;

    int num_procs, my_id;
    int len;

    char name[MPI_MAX_PROCESSOR_NAME];

    if (argc > 1) {
        n = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    chunk = ceil((float)n/nprocs);

    A = (float*) malloc(n*sizeof(float));
    B = (float*) malloc(n*sizeof(float));
    pA = (float*) malloc(chunk*sizeof(float));
    pB = (float*) malloc(chunk*sizeof(float));

    MPI_Get_processor_name(name, &len);

    printf("process %d of %d on %s\n", rank, nprocs, name);

    if (rank == 0) {
        //prepare arrays...
        for (int i = 0; i < n; i++) {
            A[i] = ((float)rand()/RAND_MAX);
            B[i] = ((float)rand()/RAND_MAX);
        }
    }

    MPI_Scatter(A, chunk, MPI_FLOAT, pA, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chunk, MPI_FLOAT, pB, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    launch_multiply(pA, pB, chunk);

    MPI_Gather(pB, chunk, MPI_FLOAT, B, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    free(A);
    free(B);
    free(pA);
    free(pB);

    MPI_Finalize();

    return 0;
}
