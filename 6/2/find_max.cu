#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.h"
#include <iostream>

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void vectorAddKernel(int n, float* deviceA, int offset) {
    unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) * offset * 2;
    if (i < n && i + offset < n)
        if (deviceA[i] < deviceA[i + offset])
            deviceA[i] = deviceA[i + offset];
}


float vectorMaxCuda(int n, float* a) {
    int threadBlockSize = 512;
    int offset = 1;

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1.0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));

    // execute kernels
    cudaEventRecord(start, 0);
    for (int i = 1; i < sqrt(n) + 1; i++) {
        vectorAddKernel<<<ceilf(ceilf(n/(float)(2 * offset))/threadBlockSize), threadBlockSize>>>(n, deviceA, offset);
        offset *= 2;
    }
    cudaEventRecord(stop, 0);

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back

    checkCudaCall(cudaFree(deviceA));

    // print the time the kernel invocation took, without the copies!
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "kernel invocation took " << elapsedTime << " milliseconds" << endl;

    return a[0];
}


int main(int argc, char* argv[]) {
    int n = atoi(argv[1]);
    timer vectorAddTimer("vector add timer");
    float* a = new float[n];
    float result = 0.0;
    float check_result = 0.0;


    srand((unsigned)time(0));

    // initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = (float)rand()/(float)RAND_MAX;
    }

    vectorAddTimer.start();
    result = vectorMaxCuda(n, a);
    vectorAddTimer.stop();

    cout << vectorAddTimer;

    // verify the resuls

    for(int i=0; i<n; i++) {
        if (a[i] > check_result) {
            check_result = a[i];
        }
    }
    if (check_result != result)
        cout << "error in results! result is " << result << ", but should be " << check_result << endl;

    cout << "results OK!" << endl;

    delete[] a;

    return 0;
}
