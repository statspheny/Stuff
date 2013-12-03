#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <cuda_runtime.h>

#define N 20000
#define GRID_D1 20
#define GRID_D2 2
#define BLOCK_D1 512
#define BLOCK_D2 1
#define BLOCK_D3 1


extern "C"
{

__global__ void 
rtruncnorm_kernel(float *vals, int n, 
                  float *mu, float *sigma, 
                  float *lo, float *hi,
                  int mu_len, int sigma_len,
                  int lo_len, int hi_len,
                  int maxtries,
		  int rng_a,    //RNG seed constant
		  int rng_b,    //RNG seed constant
		  int rng_c)    //RNG seed constant
{

    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;

    // Copied this from hello_world
    if (idx < N){  
        printf("Hello world! My block index is (%d,%d) [Grid dims=(%d,%d)], 3D-thread index within block=(%d,%d,%d) => thread index=%d\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, threadIdx.y, idx);
    } else {
        printf("Hello world! My block index is (%d,%d) [Grid dims=(%d,%d)], 3D-thread index within block=(%d,%d,%d) => thread index=%d [### this thread would not be used for N=%d ###]\n", blockIdx.x, blockIdx.y, gridDim.x, gridDim.y, threadIdx.x, threadIdx.y, threadIdx.y, idx, N);
    }


    // Setup the RNG:

    // Sample:
    
    return;


  /* // Notes/Hints from class */
  /* // i.e. threadIdx.x .y .z map these to a single index */
  /* // */
  /* // Check whether idx < N */
  /* // */
  /* // Initialize RNG */
  /* curandState rng; */
  /* curand_init(rng_a*idx+rng_b,rng_c,0,&rng); */

  /* // Sample the truncated normal */
  /* // mu for this index is mu[idx] */
  /* // sigma for this index is sigma[idx] */
  /* // a for this index is a[idx] */
  /* // b for this index is b[idx] */

  /* // X_i ~ Truncated-Normal(mu_i,sigma_i;[a_i,b_i]) */

  /* // Sample N(mu, sigma^2): */
  /* x[idx] = mu[idx] + sigma[idx]*curand_normal(&rng); */

  /* // To get the random uniform curand_uniform(&rng) */

  /* return; */

}

} // END extern "C"


int main(int argc,char **argv)
{
    const dim3 blockSize(BLOCK_D1, BLOCK_D2, BLOCK_D3);
    const dim3 gridSize(GRID_D1, GRID_D2, 1);
    int nthreads = BLOCK_D1*BLOCK_D2*BLOCK_D3*GRID_D1*GRID_D2;
    if (nthreads < N){
        printf("\n============ NOT ENOUGH THREADS TO COVER N=%d ===============\n\n",N);
    } else {
        printf("Launching %d threads (N=%d)\n",nthreads,N);
    }
    

    // Define the parameters
    float vals[10];
    int n = 10;
    int mu_len=10;
    int sigma_len=10;
    int lo_len=10;
    int hi_len=10;
    float mu[mu_len];
    float sigma[sigma_len]; 
    float lo[lo_len];
    float hi[hi_len];
    int maxtries=50;
    int rng_a=121;    //RNG seed constant
    int rng_b=132;    //RNG seed constant
    int rng_c=143;    //RNG seed constant


    // launch the kernel
    rtruncnorm_kernel<<<gridSize, blockSize>>>(vals, n, 
					       mu, sigma, lo, hi,
					       mu_len, sigma_len, lo_len, hi_len,
					       maxtries,
					       rng_a, rng_b, rng_c);
    
    // Need to flush prints...
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr){
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    } else {
        printf("kernel launch success!\n");
    }
    
    printf("That's all!\n");

    return 0;
}

