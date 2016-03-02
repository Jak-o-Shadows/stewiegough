# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:24:01 2016

@author: Jak
"""
import math
import numpy


import fk


import time



def rue(bPos, pPos, s, c, w, sigma):
    #Find distance/radius from centre of base to centre of platform
    pass
    return None
    
    
def ik(bPos, pPos, s, c, w, a):
    
    
    
    
    phi = a[3]
    th = a[4]
    psi = a[5]
    #Must translate platform coordinates into base coordinate system
    #Calculate rotation matrix elements
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(th)
    sth = math.sin(th)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)   
    #Hence calculate rotation matrix
    #Note that it is a 3-2-1 rotation matrix
    Rzyx = numpy.array([[cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi] \
                        ,[spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi] \
                        , [-sth, cth*sphi, cth*cphi]])
    #Hence platform sensor points with respect to the base coordinate system
    xbar = a[0:3] - bPos
    
    #Hence orientation of platform wrt base
    
    uvw = numpy.zeros(pPos.shape)
    for i in xrange(6):
        uvw[i, :] = numpy.dot(Rzyx, pPos[i, :])
        
    
    l_i = numpy.sqrt(numpy.sum(numpy.square(xbar + uvw),1))
    
    #arm end location on platform
    upper = pPos + (xbar+uvw)
    
    #Looking at xz plane only:
    #Hence ignore y component
    l_projected = numpy.sqrt(numpy.square(l_i) + numpy.square(l_i))
    print l_projected
    
    #angle betweern l_i & pivot arm A_i
    l_iA_i = numpy.arccos((numpy.square(l_projected) + numpy.square(s) - numpy.square(c))/(2*l_projected*c))
    
    #Angle between z axis & projected l_i
    chi = numpy.arctan2(upper[:, 0] - bPos[:, 0], upper[:, 2] - bPos[:, 2])
    print chi
    
    sigma = math.pi/2 - chi - l_iA_i
    print sigma
    
    print numpy.degrees(sigma)
    print fk(bPos, pPos, s, c, w, sigma)
    
    

def fk(bPos, pPos, s, c, w, sigma):
    """
    base attachmetn loc in base cs  -bPos
    platform attachment loc in platform cs  -pPos
    length of pivot arm  -s
    length of connecting rods  -c
    rotation about z axis of pivot arm  -w
    measured angles   -sigma
    """
    #newton-raphson
    tol_f = 1e-3;
    tol_a = 1e-3;
    #iteration limits
    maxIters = 5
    iterNum = 0
    
    #initial guess position
    # a = [x, y, z, phi, theta, psi] - angles in degrees initially
    a = [0, 0, 130, 0.1, 0.1, 0.1]
    a[3:] = [math.radians(x) for x in a[3:]] #convert to radians
    a = numpy.array(a).transpose()
    while iterNum < maxIters:
        #time.sleep(3)
        iterNum += 1
        
        
        phi = a[3]
        th = a[4]
        psi = a[5]
        #Must translate platform coordinates into base coordinate system
        #Calculate rotation matrix elements
        cphi = math.cos(phi)
        sphi = math.sin(phi)
        cth = math.cos(th)
        sth = math.sin(th)
        cpsi = math.cos(psi)
        spsi = math.sin(psi)   
        #Hence calculate rotation matrix
        #Note that it is a 3-2-1 rotation matrix
        Rzyx = numpy.array([[cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi] \
                            ,[spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi] \
                            , [-sth, cth*sphi, cth*cphi]])
        #Hence platform sensor points with respect to the base coordinate system
        xbar = a[0:3] - bPos
        
        #Hence orientation of platform wrt base
        
        uvw = numpy.zeros(pPos.shape)
        for i in xrange(6):
            uvw[i, :] = numpy.dot(Rzyx, pPos[i, :])
            
        
        l_i = numpy.sum(numpy.square(xbar + uvw),1)
            
        
               
    
        #lever arm location in base CS

        A_i = bPos + numpy.array([s*numpy.cos(w), s*numpy.sin(w), s*numpy.sin(sigma)]).transpose()

        d2 = pPos + (xbar+uvw) - A_i
        
        """
        #xz plane
        deltaSubSigma = numpy.arccos(d2[:,2]/d2[:, 1])
        delta = sigma + deltaSubSigma
        
        #yz plane
        eta = numpy.arctan2(d2[:, 1], d2[:, 2])
        """
        #Find static arm length
        c_i = numpy.sum(numpy.square(d2), 1)
        print numpy.sqrt(c_i)
        #print c
        #import sys
        #sys.exit()
            
        
        
        
        #Hence find value of objective function
        #The calculated lengths minus the actual length
        #f = -1 * (l_i - numpy.square(L))
        f = -1 * (c_i - numpy.square(c))
        sumF = numpy.sum(numpy.abs(f))
        if sumF < tol_f:
            #success!
            #print "Converged! Output is in 'a' variable"
            break
        
        #As using the newton-raphson matrix, need the jacobian (/hessian?) matrix
        #Using paper linked above:
        dfda = numpy.zeros((6, 6))
        sin = math.sin
        cos = math.cos
        alpha = phi
        beta = th
        gamma = psi
        x = a[0]
        y = a[1]
        z = a[2]
        for i in xrange(6):
            #from mathematica
            b = bPos[i]
            p = pPos[i]
            sig = sigma[i]
            s_i = s[i]
            lam = w[i]

            
            dfda[i, 0] = 2*(x-s_i*cos(lam) - 2*b[0] + p[0] + cos(alpha)*cos(beta)*p[0] + (-cos(gamma)*sin(alpha) + cos(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(alpha)*cos(gamma)*sin(beta) + sin(alpha)*sin(gamma))*p[2])            
            
            dfda[i, 1] = 2*(y-s_i*sin(lam) - 2*b[1] + cos(beta)*sin(alpha)*p[0] + p[1] + (cos(alpha)*cos(gamma) + sin(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(gamma)*sin(alpha)*sin(beta) - cos(alpha)*sin(gamma))*p[2])
            
            dfda[i, 2] = 2*(z-s_i*sin(sig) - 2*b[2] -sin(beta)*p[0] + cos(beta)*sin(gamma)*p[1] + p[2] + cos(beta)*cos(gamma)*p[2])
            
            dfda[i, 3] = 2*(y-s_i*sin(lam) - 2*b[1] + cos(beta)*sin(alpha)*p[0] + p[1] + (cos(alpha)*cos(gamma) + sin(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(gamma)*sin(alpha)*sin(beta) - cos(alpha)*sin(gamma))*p[2]) \
                        *(cos(alpha)*cos(beta)*p[0] + (-cos(gamma)*sin(alpha)+cos(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(alpha)*cos(gamma)*sin(beta) + sin(alpha)*sin(gamma))*p[2]) \
                        +2*(-cos(beta)*sin(alpha)*p[0] + (-cos(alpha)*cos(gamma) - sin(alpha)*sin(beta)*sin(gamma))*p[1] + (-cos(gamma)*sin(alpha)*sin(beta) + cos(alpha)*sin(gamma))*p[0]) \
                        *(x-s_i*cos(lam) - 2*b[0] + p[0] + cos(alpha)*cos(beta)*p[0] + (-cos(gamma)*sin(alpha) + cos(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(alpha)*cos(gamma)*sin(beta) + sin(alpha)*sin(gamma))*p[0])
            
            dfda[i, 4] = 2*(z-s_i*sin(sig) - 2*b[0] -sin(beta)*p[0] + cos(beta)*sin(gamma)*p[1] + p[2] + cos(beta)*cos(gamma)*p[2])*(-cos(beta)*p[0] -sin(beta)*sin(gamma)*p[1] - cos(gamma)*sin(beta)*p[2]) \
                        +2*(-sin(alpha)*sin(beta)*p[0] + cos(beta)*sin(alpha)*sin(gamma)*p[1] + cos(beta)*cos(gamma)*sin(alpha)*p[2])*(y-s_i*sin(lam) - 2*b[1] + cos(beta)*sin(alpha)*p[0] + p[1] + (cos(alpha)*cos(gamma) + sin(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(gamma)*sin(alpha)*sin(beta) - cos(alpha)*sin(gamma))*p[2]) \
                        +2*(-cos(alpha)*sin(beta)*p[0] + cos(alpha)*cos(beta)*sin(gamma)*p[1] + cos(alpha)*cos(beta)*cos(gamma)*p[2])*(x-s_i*cos(lam) - 2*b[0] + p[0] + cos(alpha)*cos(beta)*p[0] + (-cos(gamma)*sin(alpha) + cos(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(alpha)*cos(gamma)*sin(beta) + sin(alpha)*sin(gamma))*p[2])
            
            dfda[1, 5] = 2*(z-s_i*sin(sig) - 2*b[2] - sin(beta)*p[0] + cos(beta)*sin(gamma)*p[1] + p[2] + cos(beta)*cos(gamma)*p[2])*(cos(beta)*cos(gamma)*p[2] - cos(beta)*sin(gamma)*p[2]) \
                        +2*(x-s_i*cos(lam) - 2*b[0] + p[0] + cos(alpha)*cos(beta)*p[0] + (-cos(gamma)*sin(alpha) + cos(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(alpha)*cos(gamma)*sin(beta) + sin(alpha)*sin(gamma))*p[2]) \
                        *((cos(alpha)*cos(gamma)*sin(beta) + sin(alpha)*sin(gamma))*p[1] + (cos(gamma)*sin(alpha) - cos(alpha)*sin(beta)*sin(gamma))*p[2]) \
                        +2*(y-s_i*sin(lam) - 2*b[1] + cos(beta)*sin(alpha)*p[0] + p[1] + (cos(alpha)*cos(gamma) + sin(alpha)*sin(beta)*sin(gamma))*p[1] + (cos(gamma)*sin(alpha)*sin(beta) - cos(alpha)*sin(gamma))*p[2]) \
                        *((cos(gamma)*sin(alpha)*sin(beta) - cos(alpha)*sin(gamma))*p[1] + (-cos(alpha)*cos(gamma) - sin(alpha)*sin(beta)*sin(gamma))*p[2])

    
    
        #Hence solve system for delta_{a} - The change in lengths
        delta_a = numpy.linalg.solve(dfda, f)
    
        if abs(numpy.sum(delta_a)) < tol_a:
            #print "Small change in lengths -- converged?"
            break
        a = a + delta_a
        
    if iterNum >= maxIters:
        print "max iterations exceeded"
    
    #for i in xrange(3,6):
    #    a[i] = math.degrees(a[i])
    print "In %d iterations" % (iterNum)
    return a



class ConfigBased():
    def __init__(self):
        from configuration import *
        self.bPos = numpy.array(bPos)
        self.pPos = numpy.array(pPos)
        
        self.s_i = numpy.array([35]*6) #lever arm length
        self.c_i = numpy.array([125]*6) #rotary arm length
        self.eta_i = numpy.radians(numpy.array([-90, 90, 180, -270, 270, 0])) #rotation from x axis    
        
        
    def fk(self, angles):
        return fk(self.bPos, self.pPos, self.s_i, self.c_i, self.eta_i, numpy.radians(numpy.array(angles)))
        
    def ik(self, a):
        return ik(self.bPos, self.pPos, self.s_i, self.c_i, self.eta_i, a)

if __name__ == "__main__":
    c = ConfigBased()
    c.ik([0, 0, 100, 0, 0, 0])
    #c.fk([0, 0, 0, 0, 0, 0])
    

    
    
    
    
    

