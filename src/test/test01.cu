#include "../cuda/Communication.h"
#include <cuda_runtime.h>
#include <chrono>
#define N 10
int main(int argc, char **argv)
{
    MPI_Init(NULL, NULL);
    Communication c_send;
    Communication c_recv;
    c_send.init();
    c_recv.init();
    if (c_send.getrank() == 0)
    {
        int **send_buff, **send_buff_d;
        cudaMallocHost((void **)&send_buff, sizeof(size_t) * 1);
        cudaMallocHost((void **)&send_buff_d, sizeof(size_t) * 1);
        cudaMallocHost((void **)&send_buff[0], sizeof(int) * N);
        cudaMalloc((void **)&send_buff_d[0], sizeof(int) * N);
        for (int i = 0; i < N; i++)
        {
            send_buff[0][i] = i;
        }
        cudaMemcpy(send_buff_d[0], send_buff[0], sizeof(int) * N, cudaMemcpyHostToDevice);
        c_send.initsend(1, 1, (void **)send_buff_d);
        auto start = std::chrono::system_clock::now();
        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        while (elapsed < 15 * 1000)
        {
            // c_send.roopsend(N);
            c_send.roopsendsync(N);
            end = std::chrono::system_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
        c_send.printcounter();
        c_send.waittask();
    }
    else
    {
        void **recv_buff;
        cudaHostAlloc(&recv_buff, sizeof(size_t) * 2, cudaHostAllocDefault);
        for (int i = 0; i < 2; i++)
        {
            cudaMalloc(&recv_buff[i], sizeof(int) * N);
        }
        c_recv.initrecv(N, 0, 2, recv_buff);
        auto start = std::chrono::system_clock::now();
        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        while (elapsed < 15 * 1000)
        {
            c_recv.rooprecv();
            end = std::chrono::system_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
        c_recv.printcounter();
        c_recv.waittask();
        c_recv.printbuff();
    }
    MPI_Finalize();
}