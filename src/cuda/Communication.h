#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <cuda.h>
#if !defined(OPEN_MPI) || !OPEN_MPI
#error This source code uses an Open MPI-specific extension
#endif

/* Needed for MPIX_Query_cuda_support(), below */
#include <mpi-ext.h>

class Communication
{
public:
    // コンストラクタ
    Communication();
    // デストラクタ
    ~Communication();
    // メソッド
    void init();
    void initsend(int to_rank, void *send_buff);
    void initrecv(int max_recv_size, int form_rank);
    void roopsend(int send_size);
    void rooprecv();
    void waittask();
    int getrank();
    void printbuff();

private:
    int rank;
    int size;
    int task_finish_flag;
    int max_recv_size;
    int to_rank;   // 送信先
    int from_rank; // 受信先
    MPI_Request request;
    MPI_Status status;
    void *comm_buff;
};
#endif // COMMUNICATION_H