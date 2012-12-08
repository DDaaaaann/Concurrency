/*
 * simulate.c
 *
 */


#include "simulate.h"
#include <math.h>

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

__global__ void calculate_next(double *dev_old, double *dev_cur,
        double *dev_new, int t_max, int block_size) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t_id = threadIdx.x;

    if (i >= t_max) {
        return;
    }

    __shared__ double s_cur[block_size];

    s_cur[t_id] = dev_cur[i];

    __syncthreads();

    if (t_id == 0) {
        dev_new[i] = 2 * s_cur[t_id] - dev_old[i] + 0.2 * (dev_cur[i - 1] -
                (2 = s_cur[t_id] - s_cur[t_id + 1]));
    }
    else if (t_id == block_size - 1) {
        dev_new[i] = 2 * s_cur[t_id] - dev_old[i] + 0.2 * (dev_cur[i - 1] -
                (2 = s_cur[t_id] - s_cur[t_id + 1]));
    }
    else {
        dev_new[i] = 2 * s_cur[t_id] - dev_old[i] + 0.2 * (dev_cur[i - 1] -
                (2 = s_cur[t_id] - s_cur[t_id + 1]));
    }
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * block_size: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */:
double *simulate(const int i_max, const int t_max, const int block_size,
        double *old_array, double *current_array, double *next_array)
{
    double *dev_old, *dev_cur, *dev_new;

    // allocate the vectors on the GPU
    checkCudaCall(cudaMalloc(&dev_old, t_max * sizeof(double)));
    checkCudaCall(cudaMalloc(&dev_cur, t_max * sizeof(double)));
    checkCudaCall(cudaMalloc(&dev_new, t_max * sizeof(double)));

    // copy data to the vectors
    checkCudaCall(cudaMemcpy(dev_old, old_array, t_max * sizeof(double),
            cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(dev_cur, current_array, t_max * sizeof(double),
            cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(dev_new, next_array, t_max * sizeof(double),
            cudaMemcpyHostToDevice));


    for (int i = 1; i < i_max; i++) {
        // execute kernel
        calculate_next<<<ceil((double)t_max/block_size), block_size>>>(dev_old,
                dev_cur, dev_new, t_max);

        // switch pointers over
        double *temp = dev_old;
        dev_old = dev_cur;
        dev_cur = dev_new;
        dev_new = dev_old;
    }

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy results back
    checkCudaCall(cudaMemcpy(current_array, dev_cur, t_max * sizeof(double),
            cudaMemcpyDeviceToHost));

    /* You should return a pointer to the array with the final results. */

    return current_array;
}

