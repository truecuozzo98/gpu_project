#include <iostream>

using namespace std;

__global__ void helloFromGPU () {
    int tID = threadIdx.x;
    printf("Hello World from GPU (I am thread %d)!\n", tID);
}

void printDeviceProps() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);


        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 1);
        printf("Device Number: %d\n", 1);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
               prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");

}

int main() {
    //# hello from GPU
    cout << "Hello World from CPU!" << endl;
    cudaSetDevice(1);
    helloFromGPU <<<1, 10>>>();

    //printDeviceProps();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
    cudaDeviceSynchronize();
    return 0;
}

