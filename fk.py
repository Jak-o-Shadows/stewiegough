# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 21:10:12 2015

@author: Jak
"""

import math
import numpy
import matplotlib.pyplot as plt


def ik(bPos, pPos, a, ik=True):
    """Finds leg lengths L such that the platform is in position defined by
    a = [x, y, z, alpha, beta, gamma]
    """
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
        
    
    L = numpy.sum(numpy.square(xbar + uvw),1)
	
    #In the IK, the leg lengths are the length of the vector (xbar+uvw)
    return numpy.sqrt(L)
    
    
def platformLegPosition(pPos, a):
    """
        What is the position of each platform leg joint wrt the base CS
    """
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
    
    #Hence orientation of platform legs wrt base
    uvw = numpy.zeros(pPos.shape)
    for i in xrange(6):
        uvw[i, :] = numpy.dot(Rzyx, pPos[i, :])
        uvw[i, :] += a[0:3]
    return uvw
    

def fk(bPos, pPos, L):  
    
    #newton-raphson
    tol_f = 1e-3;
    tol_a = 1e-3;
    #iteration limits
    maxIters = 1e3
    iterNum = 0
    
    #initial guess position
    # a = [x, y, z, phi, theta, psi] - angles in degrees initially
    a = [20, 20, 100, 10, 10, 10]
    a[3:] = [math.radians(x) for x in a[3:]] #convert to radians
    a = numpy.array(a).transpose()
    while iterNum < maxIters:
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
            
        
        
        
        
        
        #Hence find value of objective function
        #The calculated lengths minus the actual length
        f = -1 * (l_i - numpy.square(L))
        sumF = numpy.sum(numpy.abs(f))
        if sumF < tol_f:
            #success!
            print "Converged! Output is in 'a' variable"
            break
        
        #As using the newton-raphson matrix, need the jacobian (/hessian?) matrix
        #Using paper linked above:
        dfda = numpy.zeros((6, 6))
        dfda[:, 0:3] = 2*(xbar + uvw)
        for i in xrange(6):
            #Numpy * is elementwise multiplication!!
            #Indicing starts at 0!
            #dfda4 is swapped with dfda6 for magic reasons!  
            dfda[i, 5] = 2*(-xbar[i,0]*uvw[i,1] + xbar[i,1]*uvw[i,0]) #dfda4
            dfda[i, 4] = 2*((-xbar[i,0]*cpsi + xbar[i,1]*spsi)*uvw[i,2] \
                            - (pPos[i,0]*cth + pPos[i,1]*sth*sphi)*xbar[i,2]) #dfda5
            dfda[i, 3] = 2*pPos[i, 1]*(numpy.dot(xbar[i,:],Rzyx[:,2])) #dfda
    
        #Hence solve system for delta_{a} - The change in lengths
        delta_a = numpy.linalg.solve(dfda, f)
    
        if abs(numpy.sum(delta_a)) < tol_a:
            print "Small change in lengths -- converged?"
            break
        a = a + delta_a
    
    #for i in xrange(3,6):
    #    a[i] = math.degrees(a[i])
    print "In %d iterations" % (iterNum)
    return a
    
    
def forces(bPos, pPos, a, M, F):
    """
        Calculates the force in each leg, given the leg geometry, platform
        position, and loading (Moments, Force at centre of platform)
    """
    
    
    
    M = numpy.array(M)
    F = numpy.array(F)
    
    uvw = platformLegPosition(pPos, a) #wsrt base CS

    #As pin joints, no moments at each joint
    #As leg position is known, the direction of each leg force is known (well, 
    #   it's in-line with the leg. +- not known)

    #For each leg, find the factor that splits the force into x, y, z directions
    #   in the platform CS


    vec = numpy.zeros(bPos.shape)
    for i in xrange(6):
        #Get direction vector of leg
        vec[i, :] = uvw[i, :] - bPos[i, :]
        #normalise it
        vec[i, :] = vec[i, :]/numpy.sqrt(numpy.sum(numpy.square(vec[i, :])))


    #Force equations
    #   These give 3 equations (one each in the cardinal directions), in the 6
    #   unknowns of the leg forces
    A = numpy.zeros((6, 6))
    for i in xrange(6):
        A[0:3, i] = numpy.transpose(vec[i, :])

    #Moment equations
    #   These give 3 equations, in the 6 unknowns of the leg forces.
    #   More complicated than the force - have to do the whole force at a
    #   distance makes a moment thing
    for i in xrange(6):
        leverArms = numpy.cross(pPos[i, :], vec[i, :])
        A[3:, i] = leverArms

    b = numpy.zeros((6, 1))
    b[0:3, 0] = F
    b[3:, 0] = M

    legForces = numpy.linalg.solve(A, b)
    return legForces

        
def centreTorque(bPos, pPos, a, legForces):
    """
        For each stewart platform leg, with force in legForces, and position
        define by bPos, pPos, a, what is the torque of the joint at the knee
        
        ie. motor is not attached to the platform or base
    
    """
    from configuration import height
    
    lowerLength = (1/3.0)*height * 1e-3
    upperLength = math.floor(1e3*calcLegLengths(bPos, pPos, a)[0][0])/1e3
    print "Legs: ", 1e3*lowerLength, 1e3*upperLength
    
    #lowerLength = 30e-3
    #upperLength = 110e-3
    
    
    
    uvw = platformLegPosition(pPos, a) #wrt base CS
    
    #Hence leg lengths are
    legs = uvw - bPos
    
    legLengthsSquared = numpy.sum(numpy.square(legs),1)
    
    angles = numpy.zeros((6, 1))
    baseAngles = numpy.zeros((6, 1))
    moments = numpy.zeros((6, 1))
    for i in xrange(6):
        #Calculate angle of motor
        cosAngle = (legLengthsSquared[i] - lowerLength**2 - upperLength**2)/(-2*lowerLength*upperLength)
        angles[i] = math.acos(cosAngle)
        #Hence calculate where the moment joint is wrt the base
        baseAngles[i] = math.asin(upperLength*math.sin(angles[i])/math.sqrt(legLengthsSquared[i]))
        radialAngle = math.atan2(bPos[i, 1], bPos[i, 0])
        dx = lowerLength*math.sin(baseAngles[i])*math.cos(radialAngle)
        dy = lowerLength*math.sin(baseAngles[i])*math.sin(radialAngle)
        dz = lowerLength*math.cos(baseAngles[i])        
        posExtra = numpy.array([dx, dy, dz])
        motorPoint = bPos[i, :] + posExtra        
        #Hence calculate lever arm of moment/force
        leverArm = numpy.cross(motorPoint - bPos[i, :], legs[i, :])
        leverArm = numpy.sqrt(numpy.sum(numpy.square(leverArm))) / numpy.sqrt(numpy.sum(numpy.square(legs[i, :])))
        #Hence calculate moment
        moments[i] = leverArm*legForces[i]

    return moments
        


    
def calcLegLengths(bPos, pPos, a):
    """
        Find a feasible lever arm and other leg joint length, based on a starting pos
        
    """
    from configuration import height
    base0 = math.radians(45)
    lowerLength = height * 1e-3
    
    uvw = platformLegPosition(pPos, a) #wrt base CS
    
    #Hence leg lengths are
    legs = uvw - bPos
    
    legLengthsSquared = numpy.sum(numpy.square(legs),1)
    
    lengths = numpy.zeros((6, 1))
    for i in xrange(6):
        lengths[i] = math.sqrt(legLengthsSquared[i] + lowerLength**2 - 2*math.sqrt(legLengthsSquared[i])*lowerLength*math.cos(base0))
        
    return lengths

    
    
    
def main():   

	#Load S-G platform configuration and convert to numpy arrays
    from configuration import *
    bPos = numpy.array(bPos)
    pPos = numpy.array(pPos)
    #Convert to m
    bPos = (1e-3)*bPos
    pPos = (1e-3)*pPos
    
    print "base joint positions"
    print bPos
    print "platform joint positions"
    print pPos

    
    #L = numpy.array([122.759, 122.759, 122.759, 122.759, 122.759, 122.759]).transpose()
    #a = fk(bPos, pPos, L)
    #print a
    #print ik(bPos, pPos, a)
    #a = numpy.array([0,0,0, 1, 0, 0]).transpose()
    #print ik(bPos, pPos, a)
    
    #Other test
#    lengths = []
#    t = numpy.arange(0, math.pi/6, 0.01)
#    for i in t:
#        a = numpy.array([0,0,0, i, 0, 0]).transpose()
#        print a
#        l = ik(bPos, pPos, a)
#        lengths.append(l)
#    angle = []
#    for L in lengths:
#        print "L", L
#        a = fk(bPos, pPos, L)
#        angle.append(a[3])
#    plt.plot(t, angle)
#    plt.xlabel('Input Angle (rad)')
#    plt.ylabel('Calculated Angle')
#    plt.show()
#    plt.plot(t, angle-t)
#    plt.xlabel('Input Angle (rad)')
#    plt.ylabel('Error (rad)')
#    plt.show()
    a = [0, 0, height*1e-3, 0, 0, 0]
    print "Acceptable upper leg lengths (mm)"
    print 1e3*calcLegLengths(bPos, pPos, a)

    
    
    M = [0, 0, 0]
    F = [0, 1*9.81, 0]
    legForces= forces(bPos, pPos, a, M, F)
    print "Leg forces (N)"
    print legForces
    print "Centre joint torque (Nm)"
    torques = centreTorque(bPos, pPos, a, legForces)
    print torques
    print "Centre joint torque (N mm)"
    print 1e3*torques
    print "Little 9g servo torque is 1.6 kg/cm, or"
    print 1.6*9.81*(10e-3), "Nm"
    print 1.6*9.81*(10e-3) * 1e3, "N mm"





if __name__ == "__main__":
    main()

