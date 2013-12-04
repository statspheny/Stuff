library(RCUDA)

# This overwrites the input values so is intended to process
# the data once and no second pases.

m = loadModule("rtruncnorm.ptx")
k = m$rtruncnorm_kernel

N = 200L
x = rnorm(N)
mu = 0
sigma = 1

n = N
mu = 0
sigma = 1
lo = -1
hi = 1
mu_len = N
sigma_len = N
lo_len = N
hi_len = N
maxtries = 50
rng_a = 1204    # RNG seed constant
rng_b = 1305    # RNG seed constant
rng_c = 2103    # RNG seed constant

cx = copyToDevice(x)
.cuda(k, cx, N, mu, sigma, lo, hi, mu_len, sigma_len, lo_len, hi_len, maxtries, rng_a, rng_b, rng_c,
      gridDim = c(20L,2L), blockDim = 64L)
i = cx[]

head(i)

summary(dnorm(x[1:100],mu,sigma) - i[1:100])
