#include "../cuda/Communication.h"
#include <cuda_runtime.h>
#define N 10
int main(int argc, char **argv)
{
    Communication c_send = new Communication();
    Communication c_recv = new Communication();
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
        c_send.roopsend(N);
    }
    else
    {
        c_recv.initrecv(N, 0);
        c_recv.rooprecv();
    }
}