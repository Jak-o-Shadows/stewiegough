# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:24:01 2016

@author: Jak
"""
import math
import numpy
import numpy as np


import time



def rotMat(roll, pitch, yaw):
    cphi = np.cos(pitch)
    sphi = np.sin(pitch)
    cth = np.cos(roll)
    sth = np.sin(roll)
    cpsi = np.cos(yaw)
    spsi = np.sin(yaw)
    #Hence calculate rotation matrix
    #Note that it is a 3-2-1 rotation matrix
    Rzyx = numpy.array([[cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi] \
                        ,[spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi] \
                        , [-sth, cth*sphi, cth*cphi]])
    return Rzyx

def ik(baseCentre, pPos, s, c, a):
    """
    Basic inverse kinematics of a stewart platform

    Does not consider actual leg mechanism

    :param baseCentre:
    :param pPos:
    :param a:


    """

    Rzyx = rotMat(a[3], a[4], a[5])
    # Hence platform centre with respect to the base coordinate system
    xbar = a[0:3] - baseCentre

    # Hence orientation of platform points wrt platform centre
    uvw = numpy.zeros(pPos.shape)
    for i in range(pPos.shape[0]):
        uvw[i, :] = Rzyx @ pPos[i, :]

    # Hence location of platform points wrt baseCentre
    upper = xbar+uvw

    return upper

def circleCircleIntersection(c1, r1, c2, r2, n):
    """
    :param c1: Centre of circle 1
    :param r1: Radius of circle 1
    :param c2: Centre of circle 2
    :param r2: Radius of circle 2
    :param n: Normal of the plane of the two circles

    Returns the intersection points of the two circles
    """
    # From stackoverflow: https://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection

    planeTangentVec = np.cross(c1-c2, n)
    t = planeTangentVec/np.sqrt(np.sum(np.square(planeTangentVec)))
    # Find the location of the midpoint of the chord
    #   connecting the two intersection points
    d2 = np.sum(np.square(c1-c2))
    h = 1/2.0 + (r1**2 -r2**2)/(2*d2)
    c_i = c1 + h*(c2-c1)
    r_i = np.sqrt(r1**2 - d2*h**2)

    p1 = c_i -t*r_i
    p2 = c_i +t*r_i

    return(p1, p2)

def circleSphereIntersection(c_c, r_c, n_c, c_s, r_s):
    """
    :param c_c: Centre of the circle
    :param r_c: Radius of the circle
    :param n_c: A normal vector of the circle
    :param c_s: Centre of the sphere
    :param r_s: Radius of the sphere

    Find the points where the circle intersects the sphere
    """

    # From stackoverflow: https://gamedev.stackexchange.com/questions/75756/sphere-sphere-intersection-and-circle-sphere-intersection

    # d = distance from sphere centre the plane cuts the sphere
    d = np.dot(n_c, c_c - c_s)
    if np.abs(d) > r_s:
        return ValueError("Circle does not intersect sphere")

    # Forms a new circle
    c_p = c_s + d*n_c
    r_p = np.sqrt(r_s**2 - d**2)

    # Problem collapses to circle-circle intersection
    if d == r_s:
        # Single point
        return (c_p, c_p)
    else:
        return circleCircleIntersection(c_c, r_c, c_p, r_p, n_c)





def legPos(bPos, pPos, s, c, legYawAngle=None):
    """

    Calculate the knee/midJoint position of each leg, and the lower leg angle
    with respect to the horizontal.

    Parameters
    ----------
    bPos:
        The position of the base joints. [6x3] array
    pPos:
        The position of the platform joints. [6x3] array
    s:
        The length of each lower leg (that connects to the base joints). [6x1]
    c:
        The length of each upper leg (that connects to the platform joints). [6x1]
    legYawAngle:
        The orientation of each lower leg - ie. the direction the base joint to the mid joint

    Returns
    -------
    midJoint:
        Position of the knee joints. [6x3] array
    leverAngles: radians
        The angle of the lower leg, from the horizontal, with zero pointing in the negative direction to the `legYawAngle`
    """
    virtualLegs = pPos - bPos
    # Virtual leg lengths
    l_i = numpy.sqrt(numpy.sum(numpy.square(virtualLegs),1))

    # Angle between virtual leg and lever
    cosAlpha = (np.square(s) + np.square(l_i) - np.square(c))/(2*s*l_i)
    alpha = np.arccos(cosAlpha)

    if legYawAngle is None:
        legYawAngle = np.arctan2(bPos[:, 0], bPos[:, 1])

    # Calculate the angle of the lower leg/lever arm
    #   The lever arm end is constrained to a circle
    #   The upper arm is constrained to a sphere about the upper pos
    #   => intersection
    #   This will return two points - want the one that leads to legs like:
    #   \
    #    \     /
    #     \   /
    #      \./
    #       rather than:
    #
    #       /.\
    #      /   \
    #     /     \
    #    /
    midJoint = np.full(pPos.shape, np.nan)
    leverAngles = np.full(legYawAngle.shape, np.nan)
    for legNum in range(pPos.shape[0]):
        #print("\n", legNum)

        planeYawAngle = legYawAngle[legNum]
        #print("plane yaw angle", planeYawAngle)

        upperCentre = pPos[legNum, :]
        lowerCentre = bPos[legNum, :]
        upperLength = c[legNum]
        lowerLength = s[legNum]
        lowerNorm = np.array([np.sin(planeYawAngle), -np.cos(planeYawAngle), 0])
        points = circleSphereIntersection(lowerCentre, lowerLength, lowerNorm, upperCentre, upperLength)
        #print(points)

        #Then calculate angle in the plane of the lever arm
        lowerTangent = rotMat(0, 0, planeYawAngle) @ np.array([1, 0, 0])
        #print("tangentVec", lowerTangent)
        # Use the dot product to calculate the angle
        angles = np.full((2,), np.nan)
        midJointAngles = np.full((2,), np.nan)
        for i, p in enumerate(points):
            p = p - bPos[legNum, :]
            pNorm = p/np.sqrt(np.sum(np.square(p)))
            #print(p)
            #print(p/np.sqrt(np.sum(np.square(p))))
            # Calculate the first angle - wrt the horizontal
            cosLeverAngle = np.dot(pNorm, lowerTangent)
            #print(cosLeverAngle)
            leverAngle = np.arccos(cosLeverAngle)
            angles[i] = leverAngle
            #print("leverAngle", np.degrees(leverAngle))

            # Choose which of the two solutions to pick
            #   Calculate the angle at the midjoint
            #       lower -> midJoint -> upper
            #   Use the dot-product
            u = pPos[legNum, :] - p
            uNorm = u/np.sqrt(np.sum(np.square(u)))
            midCosLeverAngle = np.dot(pNorm, uNorm)
            midJointAngle = np.arccos(midCosLeverAngle)
            midJointAngles[i] = midJointAngle

        midJointAngles = np.mod(midJointAngles, np.pi*2)  # Wrap to 0->360 deg
        #print(legNum, np.degrees(angles), np.degrees(midJointAngles))

        # Select the solution with the smaller angle - due to the
        #   construction of the angle, this will give us the outward
        #   facing joint we want
        pointSelectedIndex = np.argmin(midJointAngles)
        leverAngles[legNum] = angles[pointSelectedIndex]
        midJoint[legNum, :] = points[pointSelectedIndex]

    return midJoint, leverAngles

def fkWorking(bPos, pPos, legLower, legUpper, legYawAngle, legAnglesSet, translationInitialGuess, angleInitialGuess):
    """
    Solve the forward kinematics problem for a rotary stewart platform.

    The rotary stewart platform has a pure rotation joint connected
    to the base platform (defined by the `bPos`). These are connected to
    the top plate (at the `pPos` position (local frame to top plate)) by
    an an upper leg that can rotate freely by both joints.

    The lower leg is constrained to only rotate around the joint. Hence
    it has a defined angle in the xy plane (`legYawAngle`).

    The `legAnglesSet` defines the angle of rotation joint of the lower
    leg. Here 0 degrees = down, 90 = outward, 180 = upward, 270 = inward.

    Parameters
    ----------
    bPos:
        The (local to base origin) position of the base joints. [6x3] array
    pPos:
        The (local to platform) position of the platform joints. [6x3] array
    legLower:
        The length of each lower leg (that connects to the base joints). [6x1]
    legUpper:
        The length of each upper leg (that connects to the platform joints). [6x1]
    legYawAngle:
        The orientation of each lower leg - ie. the direction the base joint to the mid joint
    legAnglesSet:
        The set point of the rotary lower legs. [6x1] array or list
    translationInitialGuess:
        Initial guess for the translation of the platform. Use to force one of the possible solutions.
    translationInitialGuess:
        Initial guess for the translation of the platform. Use to force one of the possible solutions. [3x1] array or list.
    angleInitialGuess: radians
        Initial guess for the angle/rotation of the platform. Use to force one of the possible solutions. [3x1] array or list.

    returns:
    --------
    translation:
    angle:
    soln:
        Scipy optimizer solution object

    """
    def rotaryFkRootFunc(pPos, midJoint, legUpper, translationGuess, anglesGuess):
        # Hence position of platform based on guess
        upperNewGuess =  ik(np.array([0, 0, 0]), pPos, legLower, legUpper, list(translationGuess)+list(anglesGuess))

        midJointGuess, leverAnglesGuess = legPos(bPos, upperNewGuess, legLower, legUpper, legYawAngle)

        errorGuess = np.sum(np.square(midJointGuess-midJoint))
        if np.isnan(errorGuess):
            errorGuess = 10000

        #print("x:", list(translationGuess)+list(anglesGuess))
        #print(np.degrees(leverAnglesGuess) + 90)
        #print(midJointGuess)
        #print("")

        # Debug things
        # Calculate the length of the upper legs
        upperLegLengthRequiredGuess = np.sqrt(np.sum(np.square(upperNewGuess-midJoint), 1))
        # Error in leg length
        legLengthErrorSum = np.sum(np.square(legUpper - upperLegLengthRequiredGuess))

        return errorGuess

    # Hence calculate the position of the mid joints
    midJoint = np.full(pPos.shape, np.nan)
    for legNum in range(pPos.shape[0]):
    #    print("\n", legNum)
        legAngle = legAnglesSet[legNum] - np.pi/2
    #    print(f"Angle: {np.degrees(legAngle)}")

        planeYawAngle = legYawAngle[legNum]
        lowerCentre = bPos[legNum, :]
        lowerLength = legLower[legNum]

    #    print(f"Plane Yaw Angle: {np.degrees(planeYawAngle)}")

        sth = np.sin(legAngle)
        cth = np.cos(legAngle)
        v = lowerLength * np.array([cth, 0, sth])
    #    print("v", v)
        R = rotMat(0, 0, planeYawAngle)
    #    print("R")
    #    print(R)
        midPos = R @ v.T
    #    print("midpos", midPos)
    #    print(f"norm(midpos): {np.sqrt(np.sum(np.square(midPos)))}")

        midJoint[legNum, :] = midPos + lowerCentre

    # Now we have the position that the upper legs are connected to.
    #    Now we just need to fid the translation/angle of the upper
    #    platform that works for that

    def rotaryFkRootFuncWrapper(x):
        """Unpack the guess vector into arguments"""
        return rotaryFkRootFunc(pPos, midJoint, legUpper, x[:3], x[3:])

    import scipy.optimize
    soln = scipy.optimize.minimize(rotaryFkRootFuncWrapper,
        list(translationInitialGuess) + list(angleInitialGuess))

    translation = soln.x[0:3]
    angles = soln.x[3:]

    print(f"Translation is: {translation}")
    print(f"Rotation is: {np.degrees(angles)}")

    return translation, angles, soln






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
    maxIters = 1000
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
        for i in range(6):
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
        #print(numpy.sqrt(c_i))
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
        for i in range(6):
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
        print("max iterations exceeded")

    #for i in range(3,6):
    #    a[i] = math.degrees(a[i])
    print("In %d iterations" % (iterNum))
    return a


"""
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
"""

if __name__ == "__main__":
    c = ConfigBased()
    c.ik([0, 0, 100, 0, 0, 0])
    #c.fk([0, 0, 0, 0, 0, 0])








