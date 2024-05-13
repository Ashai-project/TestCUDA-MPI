/**
 * @file LoopSend.cu
 * @brief cuda対応MPIであることを確認し、cudaのhostメモリをプロセス間で送受信
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
#define N 1000

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
    int *send_buff_d, *recieve_buff_d, *send_buff_h, *recieve_buff_h;
    int recv_from, send_to;
    printf("Hostname    : %s\n", hostname);
    printf("MPI rank    : %d / %d  GPU device : %d / %d\n",
           mpirank, mpisize, gpurank, gpusize);
    cudaMalloc((void **)&send_buff_d, sizeof(int) * N);
    cudaMalloc((void **)&recieve_buff_d, sizeof(int) * N);
    cudaMallocHost((void **)&send_buff_h, sizeof(int) * N);
    cudaMallocHost((void **)&recieve_buff_h, sizeof(int) * N);
    cudaMemset(send_buff_d, mpirank, sizeof(int) * N);
    printf("success memset!\n");
    cudaMemcpy(send_buff_h, send_buff_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    recv_from = (mpirank + 1) % mpisize;
    send_to = (mpirank - 1 + mpisize) % mpisize;
    printf("MPI rank : %d send: %d recieve: %d Value: %d\n", mpirank, send_to, recv_from, send_buff_h[0]);
    auto start = std::chrono::system_clock::now();
    for (int iroop = 0; iroop < 10000; iroop++)
    {
        MPI_Request request[2];
        cudaMemcpy(send_buff_h, send_buff_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
        MPI_Isend(send_buff_h, N, MPI_INT, send_to, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(recieve_buff_h, N, MPI_INT, recv_from, 0, MPI_COMM_WORLD, &request[1]);
        MPI_Waitall(2, request, MPI_STATUS_IGNORE);
        cudaMemcpy(recieve_buff_d, recieve_buff_h, sizeof(int) * N, cudaMemcpyHostToDevice);
    }
    MPI_Finalize();
    auto end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("MPI rank    : %d / %d RValue : %d\n", mpirank, mpisize, recieve_buff_h[0]);
    std::cout << "time:" << elapsed << std::endl;
}
