library(RCUDA)

# This overwrites the input values so is intended to process
# the data once and no second pases.

m = loadModule("rtruncnorm.ptx")
k = m$rtruncnorm_kernel

N = 10000L
x = rnorm(N)

n = N
mu = rep(0,N)
sigma = rep(1,N)
lo = rep(-Inf,N)
hi = rep(-10,N)
mu_len = N
sigma_len = N
lo_len = N
hi_len = N
maxtries = 100
nullnum = -1    # This is the return value when we run out of maxtries
rng_a = 12042013    # RNG seed constant
rng_b = 13053024    # RNG seed constant
rng_c = 21031092    # RNG seed constant

gputime = system.time({
cx = copyToDevice(x)
.cuda(k, cx, N, mu, sigma, lo, hi, mu_len, sigma_len, lo_len, hi_len, maxtries, nullnum, rng_a, rng_b, rng_c,
      gridDim = c(3907L,1L), blockDim = c(16L,16L))
i = cx[]
})

head(i)


