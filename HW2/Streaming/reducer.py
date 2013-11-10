#!/usr/bin/env python
import sys

# function to format 
# list ['xlo','xhi','ylo','yhi',count] to 'xlo,xhi,ylo,yhi,count'
def getfullword(myword,mycount):
    fullword = myword[:]   # copy myword so not to make any changes
    fullword.append(str(mycount))
    fullword = ','.join(fullword)
    return fullword


# initialize
current_word = None
current_count = 0
word = None

for line in sys.stdin:

    # read the line of the form "xlo,xhi,ylo,yhi,1"
    line = line.strip()
    vals = line.rsplit(",")
    word = vals[0:4]
    count = vals[4]
    try:
        count = int(count)
    except ValueError:
        continue
    # add to count if the same word
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT
            fullword = getfullword(current_word,current_count)
            print fullword
        current_count = count
        current_word = word

# make sure to output last word
if current_word == word:
    fullword = getfullword(current_word,current_count)
    print fullword
