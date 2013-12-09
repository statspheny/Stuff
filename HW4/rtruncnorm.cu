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

      // initialize variables
      int counter = 0;     // counter for number of tries
      int keeptrying = 1;  // flag for when to stop the loop
      int useRobert = 0;   // flag to use Robert algorithm
      float newnum;    // initialize the number to be generated
      vals[idx] = CUDART_NAN_F;  // vals initializes to NaN

      // Initialize random number generator
      curandState rng;
      curand_init(rng_a*idx+rng_b,rng_c,0,&rng);


      // Determine whether to use Robert algorithm.
      // It will use the Robert's algorithm if the truncation limits are on the same side.
      float std_lo = (lo[idx] - mu[idx])/sigma[idx];
      float std_hi = (hi[idx] - mu[idx])/sigma[idx];
      if (std_lo * std_hi > 0) {
	useRobert = 1;
      }


      // sampling by truncating random normal
      if (!useRobert) {
	while ((counter < maxtries) && (keeptrying)) {
	  
	  counter = counter+1;

	  // Sample N(mu, sigma^2):
	  newnum = mu[idx] + sigma[idx]*curand_normal(&rng);

	  // if random number is within truncated space, do not reject, stop the loop
	  if ((newnum > lo[idx]) && (newnum < hi[idx]) ) {
	    keeptrying = 0;
	    vals[idx] = newnum;
	  }  //end if.  Else, try to generate another number

	}	  // end while loop
      }  // end truncating random normal algorithm


      // sampling using Robert algorithm
      if (useRobert) {

	float mu_minus;  // truncation side
	float alpha;     //
	float hitruncate;
	float tmpunif;
	float z;
	float psi;
	float tmparg;    // temporary float for holding values to put in math functions

	int negative = 0;   // flag for whether truncating positive or negative side of normal
	// we already know that std_lo and std_hi have the same sign
	if (std_lo < 0 ) {  
	  negative = 1;
	  mu_minus   = -std_hi;
	  hitruncate = -std_lo;
	} else {
	  mu_minus   = std_lo;
	  hitruncate = std_hi;
	}

	// alpha = (mu_minus + sqrtf(mu_minus^2+4))/2;
	tmparg = mu_minus*mu_minus+4;
	alpha = (mu_minus + sqrtf(tmparg))/2;
	
	while((counter < maxtries) && keeptrying) {
	  counter = counter + 1;

	  // z = mu_minus + Exp(alpha)
	  tmpunif = curand_uniform(&rng);
	  z = mu_minus - __logf(tmpunif)/alpha;

	  // get psi
	  if (mu_minus < alpha) {
	    tmparg = -(alpha-z)*(alpha-z)/2;
	    psi = __expf(tmparg);
	  } else {
	    tmparg = -(alpha-z)*(alpha-z)/2 + (mu_minus-alpha)*(mu_minus-alpha)/2;
	    psi = __expf(tmparg);
	  }

	  // accept if U < psi, and if z is within the truncation area
	  tmpunif = curand_uniform(&rng);
	  if ((tmpunif < psi) && (z < hitruncate)) {
	    if (negative) {
	      newnum = mu[idx] - z*sigma[idx];
	    } else {
	      newnum = mu[idx] + z*sigma[idx];
	    }
	    vals[idx] = newnum;
	    keeptrying = 0;
	  }

	}  // end while loop
      } // end if using Robert algorithm

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
  
    	// Sample the truncated normal
    	// mu for this index is mu[idx]
    	// sigma for this index is sigma[idx]
    	// a for this index is a[idx]
    	// b for this index is b[idx]

    	// X_i ~ Truncated-Normal(mu_i,sigma_i;[a_i,b_i])

