#include <mpi.h>
#include <cstdio>
#include <iostream>

__global__ void GPU_Kernel()
{
    printf(" GPU block  : %d / %d  GPU thread : %d / %d\n",
           blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main(int argc, char **argv)
{
    char hostname[256];
    int mpisize, mpirank, gpusize, gpurank, len;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Get_processor_name(hostname, &len);
    cudaGetDeviceCount(&gpusize);
    cudaSetDevice(mpirank % gpusize);
    cudaGetDevice(&gpurank);
    for (int irank = 0; irank < mpisize; irank++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpirank == irank)
        {
            printf("Hostname    : %s\n", hostname);
            printf("MPI rank    : %d / %d  GPU device : %d / %d\n",
                   mpirank, mpisize, gpurank, gpusize);
            // GPU_Kernel<<<2, 2>>>();
            int *send_b, *recieve_b;
            cudaMalloc((void **)&send_b, sizeof(int) * 10);
            cudaMalloc((void **)&recieve_b, sizeof(int) * 10);
            cudaMemset(send_b, 0, sizeof(int) * 10);
            // printf("GPU device : %d / %d SValue : %d\n", gpurank, gpusize, send_b[0]);
            cudaDeviceSynchronize();
            int recv_from = (mpirank + 1) % mpisize;
            int send_to = (mpirank - 1 + mpisize) % mpisize;
            MPI_Request request[2];
            MPI_Isend(send_b, 10, MPI_INT, send_to, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(recieve_b, 10, MPI_INT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
            MPI_Waitall(2, request, MPI_STATUS_IGNORE);
            int *recieve_b_h;
            cudaMallocHost((void **)&recieve_b_h, sizeof(int) * 10);
            cudaMemcpy((void *)recieve_b_h, (void *)recieve_b, sizeof(int) * 10, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            printf("GPU device : %d / %d RValue : %d\n", gpurank, gpusize, recieve_b_h[0]);
        }
    }
    MPI_Finalize();
}
