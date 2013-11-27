## BISECTION: This is a function that uses the bisection algorithm to
## find the root of a function.  It uses recursion.
##
## INPUTS: fun: The function to find the root of
##         int: The initial interval
##         tol: The tolerance of the convergence criteria
##         maxiter: Maximum number of iterations
##         debug: Flag for debugging option to print status
## OUTPUT: center: The point closest to the root when
##            stopped by tolerance or maxiteration

bisection <- function(fun,int,tol,maxiter,debug) {

  ## Get the center of the interval
  center = (int[1]+int[2])/2
  if(debug)
    print(maxiter)

  ## If there are no more iterations, stop
  if(maxiter<1) {
    print("reached max iter")
    print(center)
    return(center)
  }

  ## If the difference in the interval is within the tolerance, stop
  if(abs(int[1]-int[2])<tol)
    return(center)


  ## Do bisection
  else {

    ## Find the function value for both sides of interval and center
    low = fun(int[1])
    high = fun(int[2])
    middle = fun(center)

    if(debug) {  ## Debug
      print(sprintf("inteval: %f %f",int[1],int[2]))
      print(c(low,middle,high))
    }

    ## If the zero is betwen interval[1] and center, then do bisection
    ## from interval[1] to the center.  maxiter is one less
    if(low*middle<=0)
      return(bisection(fun,c(int[1],center),tol,maxiter-1,debug))

    ## If the zero is between interval[2] and center, the do bisection
    ## from interval[2] to the center.  maxiter is one less
    else if(middle*high<=0)
      return(bisection(fun,c(center,int[2]),tol,maxiter-1,debug))

    ## If both sides of the interval are both positive or negative,
    ## then show error.
    else {
      print("Both sides of iterval are positive or negative")
      return(NULL)
    } 
  } ## End bisection
}## End function


## NEWTONRAPHSON: This is a function that finds the root of a function
## using the Newton Raphson Algorithm
## INPUTS: fun:      the function to find the root of
##         funderiv: the derivative of the function
##         start:    the starting value
##         tol:      the tolerance of the convergence criteria
##         maxiter:  maximum number of iterations
##         debug:    a flag too print the statis
## OUTPUT: root of function fun
newtonraphson <- function(fun1,funderiv,start,tol,maxiter,debug) {
  ## initialize variables
  counter = 1
  x = start
  converged = FALSE

  while(!converged) {

    ## Newton Raphson step
    newx = x - fun1(x)/funderiv(x)

    ## Debug print each step
    if(debug) {
      print(counter)
      print(x)
      print(fun1(x))
    }
    
    counter=counter+1

    ## Check if to stop loop
    if(counter>maxiter)
      converged = TRUE
    if(abs(newx-x)<tol)
      converged = TRUE

    x = newx
  }
  return(x)
}



likelihoodderiv = function(x) {
  (2+x)^124*(x-1)^37*x^33*(197*x^2-15*x-68)
}

likelihooddoublederiv = function(x) {
  (2+x)^123*(x-1)^36*x^32*(19306*x^4-2940*x^3-13371*x^2+1088*x+2244)
}  
