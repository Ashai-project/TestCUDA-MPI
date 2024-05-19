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
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    task_finish_flag = 0;
}
void Communication::initsend(int to, void *send_buff)
{
    to_rank = to;
    comm_buff = send_buff;
}
void Communication::initrecv(int max_size, int from)
{
    max_recv_size = max_size;
    from_rank = from;
    cudaMalloc(&comm_buff, sizeof(int) * max_recv_size);
}

void Communication::roopsend(int send_size)
{
    MPI_Test(&request, &task_finish_flag, &status);
    if (task_finish_flag)
        MPI_Irecv(comm_buff, send_size, MPI_INT, to_rank, 0, MPI_COMM_WORLD, request);
}

void Communication::rooprecv()
{
    MPI_Test(&request, &task_finish_flag, &status);
    if (task_finish_flag)
        MPI_Irecv(comm_buff, max_recv_size, MPI_INT, from_rank, 0, MPI_COMM_WORLD, request);
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
    for(int i=0;i<max_recv_size;i++){
        std::cout<<"recv_buff["<<i<<"]"<< ((int *)comm_buff)[i]<<std::endl;
    }
    
}