#!/usr/bin/env python
import sys
import math

for line in sys.stdin:

    # separate x and y in input
    line = line.strip()
    instring  = line.split('\t')
    
    outstring = ""
    for num in instring:
        # floor num to first decimal point
        # this is to bin the x and y
        num = float(num)
        numround = math.floor(num*10)/10
        
        # write low num and high num to outstring
        outstring += "%.1f,%.1f," % (numround,numround+0.1) 


    # add a 1 to the end of the line
    # final outstring is "xlo,xhi,ylo,yhi,1"
    outstring += "1"
    print outstring


        
