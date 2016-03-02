# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 13:37:50 2015

@author: Jak
"""
import math
import numpy
import scipy
import matplotlib.pyplot as plt

from configuration import *

#convert lists to numpy arrays to allow matrix algebra
bPos = numpy.array(bPos)
pPos = numpy.array(pPos)
legMin = numpy.array(legMin)
legMax = numpy.array(legMax)
A = numpy.array(A)
B = numpy.array(B)

#http://www.iri.upc.edu/files/scidoc/1371-A-Linear-Relaxation-Method-for-Computing-Workspace-Slices-of-the-Stewart-Platform-.pdf

def rotationMatrix(phi, theta, psi):
    #Calculate rotation matrix elements
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)  
    #Hence calculate rotation matrix
    #Note that it is a 3-2-1 rotation matrix
    Rzyx = numpy.array([[cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi] \
                        ,[spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi] \
                        , [-sth, cth*sphi, cth*cphi]])
    return Rzyx
    
def calcPos(a):  
    phi = a[3]
    th = a[4]
    psi = a[5]
    #Must translate platform coordinates into base coordinate system
    Rzyx = rotationMatrix(phi, th, psi)
                        
    #Hence platform sensor points with respect to the base coordinate system
    xbar = a[0:3] - bPos
    
    #Hence orientation of platform wrt base
    
    uvw = numpy.zeros(pPos.shape)
    for i in xrange(6):
        uvw[i, :] = numpy.dot(Rzyx, pPos[i, :])
        
    #Hence location of platform attachment point in base coord system is
    q = xbar + uvw
    return q
    
def feasiblePoint(a, legMin, legMax):
    
    def checkJointLimits(q, l, pLim, bLim):
        bLim = math.pi/2 - bLim
        pLim = math.pi/2 - pLim
        """Check if any of the joint limits are past that allowed in:
            pLim - platform Limits
            bLim - base limits
            
            --Assumes that limits are with respect to perpendicular the
            platform/base
        """
        
        j_B = numpy.array([0,0,1]*6)
        j_B.shape = (6,3)
        j_P = numpy.array([0,0,1]*6)
        j_P.shape = (6,3)
        print bLim
        print numpy.cos(bLim)

        if (numpy.sum(j_B*q, 1) < l*numpy.cos(bLim).transpose()).all():
            phi = a[3]
            th = a[4]
            psi = a[5]
            rotMatT = rotationMatrix(phi, th, psi).transpose()
            qRotated = numpy.zeros((6,3))
            for i in xrange(6):
                qRotated[i, :] = numpy.dot(rotMatT, q[i, :])
            if (numpy.sum(j_P*qRotated,1) < l*numpy.cos(pLim).transpose()).all():
                return True
        return False
            
    q = calcPos(a)
    l = numpy.sqrt(numpy.sum(numpy.square(q) ,1))
    if (l > legMin).all() and (l< legMax).all():
        return checkJointLimits(q, l, B, A)
    else:
        return False

def bruteForce(xstep, zstep, ystep, legMin, legMax, lim):
    #Assume symmetrical platform

    
    x = lim
    y = lim
    z = lim

        
    feasibleBoundary = []
    
    xMax = 0
    yMax = 0
    zMax = 0
    
    for i in xrange(2*lim/xstep):
        y = lim
        for j in xrange(2*lim/ystep):
            z = lim
            for k in xrange(2*lim/zstep):
                a = [x, y, z, 0, 0, 0]
                feasible = feasiblePoint(a, legMin, legMax)
                #print a, feasible
                if feasible:
                    print "FEASIBLE POINT", x, y, z
                    feasibleBoundary.append((x, y, z))
                    if xMax < i:
                        xMax = i
                    if yMax < j:
                        yMax = j
                    if zMax < k:
                        zMax = k             
                    break;
                z -= zstep
            print y
            y -= ystep
        x -= xstep
        
    print xMax
    print yMax
    print zMax
        
    return feasibleBoundary
    
if __name__ == "__main__":
    xstep = 4
    ystep = 4
    zstep = 4
    lim = 200
    boundary = bruteForce(xstep, ystep, zstep, legMin, legMax, lim)
    boundary += bruteForce(-xstep, -ystep, -zstep, legMin, legMax, -lim)
    print len(boundary)
    import pprint
    pprint.pprint(boundary)
    x = [c[0] for c in boundary]
    y = [c[1] for c in boundary]
    z = [c[2] for c in boundary]
    import matplotlib
    xi = numpy.array(xstep*range(50))
    yi = numpy.array(ystep*range(50))
    X, Y = numpy.meshgrid(xi, yi)
    Z = matplotlib.mlab.griddata(x, y, z, xi, yi)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    #ax.plot_surface(X, Y, Z)
    #ax.contour(x, y, z) # z must be a 2D array - colour?
    plt.show()
    fig.savefig('test.png')
    
    








#
##Following paper - eqn 5ish
#m = (legMax + legMin)/2 #Mid point of allowed leg interval
#h = (legMax - legMin)/2 #half-range of allowed leg interval
#
##x = [l, g, t, d, c_alpha, c_beta, c_gamma, s_alpha, s_beta, s_gamma]
##x2 = x + x.^2
##x = x2 + [l*g, l*t, l*d, l*c_a, l*c_b, l*c_g, l*s_a, l*s_b, l*s_g
##x = x + [      g*t, g*d, g*c_a, g*c_b, g*c_g, g*s_a, g*s_b, g*s_g]
##etc
#
##bounds for each variable
#l = [legMax, legMin]
#d = [-h, h]
#t = [-math.sqrt(legMax - legMax*math.cos(A)), math.sqrt(legMax - legMax*math.cos(A))] 
#g = [-math.sqrt(legMax - legMax*math.cos(B)), math.sqrt(legMax - legMax*math.cos(A))]
##all angular limits:
#c = [-1, 1]
#
#"""Have Equations
#
#l_i = ||q_i||
#
#adding slack variable d_i
#CONSTRAINT 1
#(l_i - m_i)^2 + d_i^2 = h_i^2
#
#%joint limits
#Base joint limits:
#    joint limit on UV joint of A_i
#    j_Bi = normal vector of UV joint i
#    Thus
#    j_Bi * q_i >= l_i*cos{A_i} %Using dot product
#    Adding slack variables t_i
#    CONSTRAINT 2
#    j_Bi * q_i - l_i*cos{alpha_i} = t_i^2
#Similarly for platform joint limits:
#    B_i = angular misalignment of platform joint i
#    j_Pi = dir. vector along axis of symmetry at platform joint i, expressed in P frame
#    hence, adding slack variable g_i
#    CONSTRAINT 3
#    j_Pi*(Rzyx' * q_i) - l_i*cos{B_i} = g_i^2
#Thus workspace is set of all possible tuples:
#    W = (x, y, z, phi, theta, psi) that satisfiy the constraints
#    
#    
#Only really care about the boundary of the workspace though
#-we hit a limit - leg length, or a joint
#--> one of d_i, t_i, g_i = 0 
#--> product(1, 6, d_i * t_i * g_i) = 0
#
#
#Letting the terms of the R matrix, cos(T) & sin(T) be:
#    c_{T} = \cos{T}
#    s_{T} = \sin{T}
#Note that these have constraints, due to coming from trig formulae:
#    c_{T}^2 + s_{T}^2 = 1
#    and
#    c_{T}, s_{T} \in \left[-1, 1\right]
#    
#
#"""
#
#
#
#
#
#



