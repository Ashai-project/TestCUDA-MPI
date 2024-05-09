    
    
    int ngpus;
    cudaGetDeviceCount(&ngpus);
    
    for (int igpu = 0; igpu < ngpus; igpu++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, igpu);
        printf("Using Device %d : %s\n", igpu, deviceProp.name);

        // igpuのデバイスでデバイスコードkernelが実行される
        cudaSetDevice(igpu);
        kernel<<<grid, block>>>(...);
    }
