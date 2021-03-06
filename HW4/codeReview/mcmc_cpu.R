library(truncnorm)
library(mvtnorm)
library(MASS)

## rtruncnormR = function(N, mu, sigma, lo, hi, mu_len, sigma_len, lo_len, hi_len, 
## 	      maxtries, nullnum, rng_a, rng_b, rng_c) 
## {
##   normvalues = rnorm(N,mu,sigma)
##   truncated = normvalues[normvalues > lo & normvalues < hi]
## }

    
probit_mcmc_cpu = function(y, X, beta_0, Sigma_0_inv, niter, burnin) {

    ## Define posterior beta
    posterior_beta = numeric(niter)
    betat = t(as.matrix(beta_0))

    p = length(betat)
    allbetas = matrix(0,niter+burnin,p)
    X = as.matrix(X)
    
    N = length(y)
    lo = rep(0,N)
    lo[y==0] = -Inf
    hi = rep(0,N)
    hi[y==1] = Inf

    xinv = ginv(X)

    ## iterate niter times
    for(idx in 1:(niter+burnin)) {
        
        ## get z_i
        xTb = X %*% t(betat)
        zi = rtruncnorm(N,lo,hi,mean=xTb,sd=1)

        ## update beta
        betamean = xinv %*% zi
        betasd   = diag(1,length(betamean))
        betat = rmvnorm(1,betamean,betasd)

        if(idx %% 200 == 0) {
            print(paste("at iteration",idx,"..."))
		}

        allbetas[idx,] = betat
    }
    
    # remove the burnin
    allbetas = allbetas[(burnin+1):(burnin+niter),]
    return(allbetas)
    
}


