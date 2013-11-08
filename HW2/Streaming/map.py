# import math to use floor function

import math

# set up input and output files
infile = open("mini_out.txt","r")
outfile = open("mini_out_truncated.txt","w")


for line in infile:

    # separate x and y
    line = line.strip()
    instring  = line.split('\t')

    for num in instring:
        # floor num to first decimal point
        num = float(num)
        numround = math.floor(num*10)/10
        
        # write to outfile with tabs in between
        outstring = "%.1f" % numround 
        outfile.write(outstring)
        outfile.write("\t")

    # add a 1 and \n to each line
    outfile.write("1\n")

# close files
infile.close()
outfile.close()

print("DONE!")

        
