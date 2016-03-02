# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 21:39:48 2015

@author: Jak
"""

import math
import pprint

def sweep(a, pos, maxVal, minVal, numSteps):
    """Sweep between maxVal and minVal, taking numSteps between minVal & maxVal
    Note that 2*numSteps will be taken, as start & finish at 0
    minVal <0
    maxVal >0
    """  
    valStep = (maxVal - minVal)/numSteps
    aList = []
    #step up to maxVal
    for i in xrange(numSteps/2):
        a[pos] = valStep*i
        aList.append(list(a))
    #step back down to minVal
    for i in xrange(numSteps):
        a[pos] = maxVal - valStep*i
        aList.append(list(a))
    #go back to zero
    for i in xrange(numSteps/2 + 1):
        a[pos] = minVal + valStep*i
        aList.append(list(a))
    return aList

def writeList(l):
    l = [",".join([str(x) for x in a]) + "\n" for a in l]
    with open("positions", "wb") as f:
        f.writelines(l)


a = [0, 0, 0, 0, 0, 0]

maxAngle = math.pi/6
minAngle = -maxAngle

#Sweep angles
aList = sweep(a, 3, maxAngle, minAngle, 100)
aList += sweep(a, 4, maxAngle, minAngle, 100)
aList += sweep(a, 5, maxAngle, minAngle, 100)
#Sweep position
aList += sweep(a, 0, 0.4, -0.4, 100)
aList += sweep(a, 1, 0.4, -0.4, 100)
aList += sweep(a, 2, 0.4, -0.4, 100)

pprint.pprint(aList)
writeList(aList)






