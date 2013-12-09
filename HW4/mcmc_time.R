datalist = c("mini_data.txt","data_01.txt","data_02.txt","data_03.txt","data_04.txt","data_05.txt")

cputimes = list()
gputimes = list()

for (i in 1:6) {
  data = read.table(datalist[i],header=TRUE)
  p = ncol(data)-1
  y = data$y

  X = data[,2:(p+1)]
  beta_0 = rep(0,p)
  Sigma_0_inv = matrix(0,p,p)

  print(datalist[i])

  cputimes[[i]] = system.time({
    cpuans = probit_mcmc_cpu(y,X,beta_0,Sigma_0_inv,2000,500)
  })

  gputimes[[i]] = system.time({
    gpuans = probit_mcmc_gpu(y,X,beta_0,Sigma_0_inv,2000,500)
  })

  name = paste("mcmctimes",i,".RData",sep="")
  save.image(name)
}
