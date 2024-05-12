/**
 * @file LoopSend4.cu
 * @author Ashai-project
 * @brief GPUDirect RDMA
 * cudaデバイス上のメモリ領域からcudaデバイスのメモリ領域へRDMA
 * @date 2024-05-10
 *
 */
#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <cuda.h>
#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include <mpi-ext.h>

int main(int argc, char **argv)
{
    char hostname[256];
    int mpisize, mpirank, gpusize, gpurank, len;
    MPI_Init(&argc, &argv);
    // check cuda aware
    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does not have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support())
    {
        printf("This MPI library has CUDA-aware support.\n");
    }
    else
    {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else  /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Get_processor_name(hostname, &len);
    cudaGetDeviceCount(&gpusize);
    cudaSetDevice(mpirank % gpusize);
    cudaGetDevice(&gpurank);
    int *send_buff_d, *send_buff_h;
    int send_to;
    for (int irank = 0; irank < mpisize; irank++)
    {
        // MPI_Barrier(MPI_COMM_WORLD); // グループのプロセス間でここで同期を取る
        if (mpirank == irank)
        {
            printf("Hostname    : %s\n", hostname);
            printf("MPI rank    : %d / %d  GPU device : %d / %d\n",
                   mpirank, mpisize, gpurank, gpusize);
            // GPU_Kernel<<<2, 2>>>();
            cudaMalloc((void **)&send_buff_d, sizeof(int) * 10);
            cudaMallocHost((void **)&send_buff_h, sizeof(int) * 10);
            cudaMemset(send_buff_d, mpirank, sizeof(int) * 10);
            printf("success memset!\n");
            cudaMemcpy(send_buff_h, send_buff_d, sizeof(int) * 10, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            send_to = mpirank+4;
            MPI_Request request[1];
            MPI_Isend(send_buff_d, 10, MPI_INT, send_to, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Waitall(2, request, MPI_STATUS_IGNORE);
            cudaDeviceSynchronize();
        }
    }
    MPI_Finalize();
}
