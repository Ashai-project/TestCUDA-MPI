/**
 * @file Communication.cu
 * @brief 通信の完了をフラグとして次の通信を開始する
 * mainのはじめにMPI_Init(NULL, NULL);を呼び出すのを忘れないように
 * 一つのインスタンスで送受信はしないこと,必ず送信か受信だけ
 * @version 0.1
 * @date 2024-05-18
 *
 * @copyright Copyright (c) 2024
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
    use_buff = 0;
}
/**
 * @brief
 *
 * @param to
 * 送信先
 * @param num
 * 送信用バッファの数、ダブルバッファなら2
 * @param send_buffer
 * 送信用バッファのポインタ
 */
void Communication::initsend(int to, int num, void **send_buffer)
{
    to_rank = to;
    send_buff = send_buffer;
    buff_num = num;
    MPI_Isend(send_buff[0], 1, MPI_INT, to_rank, 0, MPI_COMM_WORLD, &request);
    std::cout << "rank: " << rank << " send to: " << to_rank << std::endl;
    counter = 0;
}
/**
 * @brief
 *
 * @param max_size
 * 受信サイズ、int最大個数
 * @param from
 * 受信先
 * @param num
 * 受信用バッファの数
 * @param recv_buff_array
 * 受信用バッファのポインタ
 */
void Communication::initrecv(int max_size, int from, int num, void **recv_buff_array)
{
    max_recv_size = max_size;
    from_rank = from;
    buff_num = num;
    recv_buff = recv_buff_array;
    // cudaHostAlloc(&recv_buff, sizeof(size_t) * buff_num, cudaHostAllocDefault);
    // for (int i = 0; i < buff_num; i++)
    // {
    //     cudaMalloc(&recv_buff[i], sizeof(int) * max_recv_size);
    // }
    MPI_Irecv(recv_buff[use_buff], max_recv_size, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &request);
    std::cout << "rank: " << rank << " recv from: " << from_rank << " buff num: " << buff_num << std::endl;
    counter = 0;
}

/**
 * @brief
 * sendのバッファが再利用可能になるまで次のsendをブロックする
 * @param send_size
 * 送信サイズ intの個数分
 */
void Communication::roopsend(int send_size)
{
    MPI_Test(&request, &task_finish_flag, &status);
    if (task_finish_flag)
    {
        MPI_Isend(send_buff[use_buff], send_size, MPI_INT, to_rank, 0, MPI_COMM_WORLD, &request);
        counter++;
        use_buff = (use_buff + 1) % buff_num;
    }
}
/**
 * @brief
 * Ssendは受信を確認するまで実行されたスレッドにwaitをかける
 * よってこの関数は受信側と完全同期した送信が行われる
 * @param send_size
 * 送信サイズ intの個数分
 */
void Communication::roopsendsync(int send_size)
{

    MPI_Ssend(send_buff[use_buff], send_size, MPI_INT, to_rank, 0, MPI_COMM_WORLD);
    counter++;
    use_buff = (use_buff + 1) % buff_num;
}

/**
 * @brief 前回の受信の終了を確認し新規の非同期受信を開始
 *
 */
void Communication::rooprecv()
{
    MPI_Test(&request, &task_finish_flag, &status);
    if (task_finish_flag)
    {
        MPI_Irecv(recv_buff[use_buff], max_recv_size, MPI_INT, from_rank, 0, MPI_COMM_WORLD, &request);
        counter++;
        use_buff = (use_buff + 1) % buff_num;
    }
}
/**
 * @brief
 * requestが完了するまでスレッド待機
 */
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