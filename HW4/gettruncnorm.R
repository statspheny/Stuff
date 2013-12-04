library(RCUDA)

# This overwrites the input values so is intended to process
# the data once and no second pases.

m = loadModule("rtruncnorm.ptx")
k = m$rtruncnorm_kernel

N = 1e6L
x = rnorm(N)
mu = 0
sigma = 1

n = N
mu = 0
sigma = 0 
lo = -1
hi = 1
mu_len = length(mu)
sigma_len = length(sigma)
lo_len = length(lo)
hi_len = length(hi)
maxtries = 50
rng_a = 12042013    # RNG seed constant
rng_b = 13053024    # RNG seed constant
rng_c = 21031092    # RNG seed constant

cx = copyToDevice
.cuda(k, cx, n, mu, sigma, lo, hi, mu_len, sigma_len, lo_len, hi_len, maxtries, rng_a, rng_b, rng_c,
      gridDim = c(64L,32L), blockDim = 512L)
i = cx[]

head(i)

summary(dnorm(x[1:100],mu,sigma) - i[1:100])
