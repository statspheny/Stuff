
mini <- FALSE

#============================== Setup for running on Gauss... ==============================#

args <- commandArgs(TRUE)

cat("Command-line arguments:\n")
print(args)

####
# sim_start ==> Lowest possible dataset number
###

###################
sim_start <- 1000
###################

if (length(args)==0){
  sim_num <- sim_start + 1
  sim_seed <- 121231
} else {
  # SLURM can use either 0- or 1-indexing...
  # Lets use 1-indexing here...
  sim_num <- sim_start + as.numeric(args[1])
  sim_seed <- (762*(sim_num-1) + 121231)
}

cat(paste("\nAnalyzing dataset number ",sim_num,"...\n\n",sep=""))

# Find r and s indices:
s <- 5
r <- 50

if (length(args)==0) {
  r_index <- 0
  s_index <- 0
} else {
  job_num <- as.numeric(args[1])
  r_index <- (job_num-1) %% r + 1
  s_index <- (job_num-1) %/% r + 1
}

set.seed(121231+s_index)

#============================== Run the simulation study ==============================#

# Load packages:
library(BH)
library(bigmemory.sri)
library(bigmemory)
library(biganalytics)

# I/O specifications:
datapath <- "/home/pdbaines/data/"
# datapath <- "data/"
outpath <- "output/"

# mini or full?
if (mini){
	rootfilename <- "blb_lin_reg_mini"
	outpath <- "output_mini/"
} else {
	rootfilename <- "blb_lin_reg_data"
	outpath <- "output/"
}

# Filenames:
infile <- paste0(datapath,rootfilename,".desc")
  
# Set up I/O stuff:

# Attach big.matrix :
z = attach.big.matrix(infile)

# Remaining BLB specs:
gamma = 0.7

n = nrow(z)
b = round(n^gamma)

# Extract the subset:
extracted.indices <- sample(1:n,b)
extracted.data <-  z[extracted.indices,]

# Reset simulation seed:
set.seed(sim_seed)

# Bootstrap dataset:
weights <- rmultinom(1,n,rep(1,b))

# Fit lm:
p <- ncol(z)-1
y <- extracted.data[,p+1]
x <- extracted.data[,1:p]

fit <- lm(y~x,weights=weights)

# Output file:
outfile <- paste0(outpath,"coef_",sprintf("%02d",s_index),"_",sprintf("%02d",r_index),".txt")

# Save estimates to file:
write(fit$coeff,outfile,1)


