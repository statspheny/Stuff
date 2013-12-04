#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include <cuda_runtime.h>


extern "C"
{

__global__ void 
rtruncnorm_kernel(float *vals, int n, 
                  float *mu, float *sigma, 
                  float *lo, float *hi,
                  int mu_len, int sigma_len,
                  int lo_len, int hi_len,
                  int maxtries,
		  int rng_a,   // RNG seed constant
		  int rng_b,   // RNG seed constant
		  int rng_c)   // RNG seed constant
{

    // Usual block/thread indexing...
    int myblock = blockIdx.x + blockIdx.y * gridDim.x;
    int blocksize = blockDim.x * blockDim.y * blockDim.z;
    int subthread = threadIdx.z*(blockDim.x * blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
    int idx = myblock * blocksize + subthread;

    	// Notes/Hints from class
    	// i.e. threadIdx.x .y .z map these to a single index
    	//
    	// Check whether idx < N
    if(idx < n) {
    	// Initialize RNG
    	curandState rng;
    	curand_init(rng_a*idx+rng_b,rng_c,0,&rng);

    	// Sample the truncated normal
    	// mu for this index is mu[idx]
    	// sigma for this index is sigma[idx]
    	// a for this index is a[idx]
    	// b for this index is b[idx]

    	// X_i ~ Truncated-Normal(mu_i,sigma_i;[a_i,b_i])

    	// Sample N(mu, sigma^2):
    	// x[idx] = mu[idx] + sigma[idx]*curand_normal(&rng);
    	vals[idx] = curand_normal(&rng);

    	// To get the random uniform curand_uniform(&rng)


    // Setup the RNG:

    // Sample:
    }
	
    return;
}

} // END extern "C"

