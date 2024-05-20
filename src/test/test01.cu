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
        int *send_buff;
        cudaMallocHost((void **)&send_buff, sizeof(int) * N);
        for (int i = 0; i < N; i++)
        {
            send_buff[i] = i;
        }
        c_send.initsend(1, send_buff);
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
        c_recv.initrecv(N, 0, 2);
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