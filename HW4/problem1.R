library(truncnorm)

source("utility.R")

DOGPU = FALSE  # flag for whether to run on personal computer or on AWS
if(DOGPU) {
  library(RCUDA)
  m = loadModule("rtruncnorm.ptx")
  k = m$rtruncnorm_kernel
}
  
## set up constants
maxtries = 100
nullnum = -1    # This is the return value when we run out of maxtries
rng_a = 12042013    # RNG seed constant
rng_b = 13053024    # RNG seed constant
rng_c = 21031092    # RNG seed constant

## TN(2,1,0,1.5)
N = 10000L
mu = rep(2,N)
sigma = rep(1,N)
lo = rep(0,N)
hi = rep(1.5,N)
mu_len = N
sigma_len = N
lo_len = N
hi_len = N

if(DOGPU) {
  x = rnorm(N)
  gputime = system.time({
    cx = copyToDevice(x)
    .cuda(k, cx, N, mu, sigma, lo, hi, mu_len, sigma_len, lo_len, hi_len, maxtries, nullnum, rng_a, rng_b, rng_c,
          gridDim = c(3907L,1L), blockDim = c(16L,16L))
    gpu1 = cx[]
  })
  png("histogram1c.png")
  hist(gpu1)
  dev.off()
}

print("finished TN(2,1,0,1.5)")

cputime1 = system.time({cpu1 = rtruncnorm(N,mean=mu,sd=sigma,a=lo,b=hi)})

# function get_cpu_time
get_cpu_time = function(n,mu0,sigma0,lo0,hi0) {
  cputime = system.time({cpu2 = rtruncnorm(n,mean=mu0,sd=sigma0,a=lo0,b=hi0)})
  return(cputime)
}

# function get_gpu_time
if(DOGPU) {
  get_gpu_time = function(n,mu0,sigma0,lo0,hi0) {
    x = rnorm(n)
    gridblockdims = compute_grid(n)
    gputimecopy1 = system.time({
      cx = copyToDevice(x)
    })
    gputimekernel = system.time({
      .cuda(k, cx, n, mu0, sigma0, lo0, hi0, n, n, n, n, maxtries, nullnum, rng_a, rng_b, rng_c,
            gridDim = gridblockdims[[1]], blockDim = gridblockdims[[2]])
    })
    gputimecopy2 = system.time({
      gpu2 = cx[]
    })
    gputime = list("gputimecopy1"=gputimecopy1,"gputimekernel"=gputimekernel,"gputimecopy2"=gputimecopy2)
    return(gputime)
  }
}

## Time GPU vs CPU time for truncated normal, for N = 10^k where k is 1:8
cputimes = list()
gputimes = list()
maxk = 8
singlemu = 2
singlesigma = 1
singlelo = 0
singlehi = 1.5

for (k in 1:maxk) {
  n = 10^k
  muall = rep(singlemu,n)
  sigmaall = rep(singlesigma,n)
  loall = rep(singlelo,n)
  hiall = rep(singlehi,n)
  cputimes[[k]] = get_cpu_time(n,muall,sigmaall,loall,hiall)
  print(paste("at k=",k,"cputime is "))
  print(cputimes[[k]])
  if(DOGPU) {
    gputimes[[k]] = get_gpu_time(n,muall,sigmaall,loall,hiall)
    print(paste("at k=",k,"gputime is "))
    print(gputimes[[k]])
  }
}

