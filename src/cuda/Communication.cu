/**
 * @file Communication.cu
 * @brief 通信の完了をフラグとして次の通信を開始する
 * @version 0.1
 * @date 2024-05-18
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "Communication.h"

void Communication::init()
{
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    task_finish_flag = 0;
}
void Communication::initsend(int to, void *send_buff)
{
    to_rank = to;
    send_buff = send_buff;
    MPI_Isend(send_buff, 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD, &request);
    std::cout << "rank: " << rank << " send to: " << to_rank << std::endl;
    counter = 0;
}
void Communication::initrecv(int max_size, int from, int num)
{
    max_recv_size = max_size;
    from_rank = from;
    buff_num = num;
    use_buff = 0;
    cudaHostAlloc(&recv_buff, sizeof(size_t) * num, cudaHostAllocDefault);
    for (int i = 0; i < buff_num; i++)
    {
        cudaMalloc(&recv_buff[i], sizeof(int) * max_recv_size);
    }
    MPI_Irecv(recv_buff[use_buff], max_recv_size, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &request);
    std::cout << "rank: " << rank << " recv from: " << from_rank << std::endl;
    counter = 0;
}

void Communication::roopsend(int send_size)
{
    MPI_Test(&request, &task_finish_flag, &status);
    if (task_finish_flag)
    {
        MPI_Isend(send_buff, send_size, MPI_INT, to_rank, 0, MPI_COMM_WORLD, &request);
        counter++;
        // std::cout << "send " << std::endl;
    }
}

void Communication::rooprecv()
{
    MPI_Test(&request, &task_finish_flag, &status);
    if (task_finish_flag)
    {
        MPI_Irecv(send_buff, max_recv_size, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &request);
        counter++;
        use_buff = (use_buff + 1) % buff_num;
        // std::cout << "recv " << std::endl;
    }
}

void Communication::waittask()
{
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

int Communication::getrank()
{
    return rank;
}

void Communication::printbuff()
{
    int *recv_h;
    cudaHostAlloc((void **)&recv_h, sizeof(int) * max_recv_size, cudaHostAllocDefault);
    cudaMemcpy(recv_h, recv_buff[0], sizeof(int) * max_recv_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < max_recv_size; i++)
    {
        std::cout << "recv_buff[" << i << "]" << recv_h[i] << std::endl;
    }
    cudaFreeHost(recv_h);
}

void Communication::printcounter()
{
    std::cout << "roop counter:" << counter << std::endl;
}