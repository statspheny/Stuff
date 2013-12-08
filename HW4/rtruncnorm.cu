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
		  float nullnum,
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

      // initialize
      int counter = 0;
      int keeptrying = 1;
      float newnum = 0;
      vals[idx] = CUDART_NAN_F;

      while ((counter < maxtries) && (keeptrying)) {

	counter = counter+1;
    	// Initialize RNG
    	curandState rng;
    	curand_init(rng_a*idx+rng_b+counter,rng_c,0,&rng);

    	// Sample the truncated normal
    	// mu for this index is mu[idx]
    	// sigma for this index is sigma[idx]
    	// a for this index is a[idx]
    	// b for this index is b[idx]

    	// X_i ~ Truncated-Normal(mu_i,sigma_i;[a_i,b_i])

    	// Sample N(mu, sigma^2):
    	newnum = mu[idx] + sigma[idx]*curand_normal(&rng);

	// if random number is within truncated space, do not reject, stop the loop
	if ((newnum > lo[idx]) && (newnum < hi[idx]) ) {
	  keeptrying = 0;
	  vals[idx] = newnum;
	}  //end if

	//  if (counter < maxtries) {
	//     keeptrying = 0;
	//  }
      }	  // end while loop

      // if reached maxtries without getting a value, then use Robert algorithm
      counter = 0;
      while((counter < maxtries) && keeptrying) {


      }  // end while loop for Robert algorithm

      // debugging purposes
      // vals[idx] = (float) counter;
      // vals[idx] = newnum;

    } // end if(idx<n)
	
    return;
}  // end kernel

} // END extern "C"


  // Other notes
  // To get the random uniform curand_uniform(&rng)
  // Setup the RNG:
  // Sample:
  
