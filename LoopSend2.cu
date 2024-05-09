#include <mpi.h>

int main(int argc, char* argv[]) {
  // Initialize MPI and CUDA
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Allocate memory on GPU
  int device;
  cudaGetDevice(&device);
  cudaDeviceSetCurrent(device);

  float* data_gpu;
  cudaMalloc(&data_gpu, sizeof(float) * 1024);

  // Initialize data on GPU
  for (int i = 0; i < 1024; ++i) {
    data_gpu[i] = rank * i;
  }

  // Send GPU memory to the previous rank
  if (rank > 0) {
    MPI_Send(data_gpu, 1024, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
  }

  // Receive GPU memory from the next rank
  if (rank < size - 1) {
    float* recv_data_gpu;
    cudaMalloc(&recv_data_gpu, sizeof(float) * 1024);

    MPI_Recv(recv_data_gpu, 1024, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Verify received data
    for (int i = 0; i < 1024; ++i) {
      if (recv_data_gpu[i] != (rank + 1) * i) {
        printf("Error: Received data does not match expected value\n");
        break;
      }
    }

    cudaFree(recv_data_gpu);
  }

  // Free GPU memory
  cudaFree(data_gpu);

  // Finalize MPI and CUDA
  MPI_Finalize();

  return 0;
}
