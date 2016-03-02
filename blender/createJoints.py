# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:30:25 2015

Create joints

@author: Jak
"""




import bge

#name configuration
#legs will be: leg1Upper, leg1Upper, etc
#leg1Lower, leg2Lower, etc
legPrefix = "leg"
legUpper = "Upper"
legLower = "Lower"
numLegs = 6

platformName = "platform"
baseName = "base"

#Add the constraints - don't edit below here
objects = bge.logic.getCurrentScene().objects
platform = objects[platformName].getPhysicsId()
base = objects[baseName].getPhysicsId()

#Axis labels, because they are integers
aLab = {}
aLab["x"] = 0
aLab["y"] = 1
aLab["z"] = 2
aLab["xr"] = 3 #x axis rotation
aLab["yr"] = 4
aLab["zr"] = 5

for i in range(numLegs):
    i+=1
    #add joint between leg parts first
    upper = objects[legPrefix+str(i)+legUpper].getPhysicsId()
    lower = objects[legPrefix+str(i)+legLower].getPhysicsId()
    pos = (0,0,0)
    axis = (0,0,0)
    disableCollisions = 128
    #Create constraint 1
    joint = bge.constraints.createConstraint(upper, lower, \
                                    bge.constraints.GENERIC_6DOF_CONSTRAINT,\
                                    pos[0], pos[1], pos[2], axis[0], axis[1],\
                                    axis[2], disableCollisions)
    #Set joint limits
    #if min = max, locked
    #if min < max, constrained to within those
    #if min > max, unconstrained
    joint.setParam(aLab["x"], 0,0)
    joint.setParam(aLab["y"], 0,0)
    joint.setParam(aLab["z"], 1,0) #unconstrained
    joint.setParam(aLab["xr"], 0,0)
    joint.setParam(aLab["yr"], 0,0)
    joint.setParam(aLab["zr"], 0,0)
    #create constraint 2
    #Create constraint 1
    joint = bge.constraints.createConstraint(lower, upper, \
                                    bge.constraints.GENERIC_6DOF_CONSTRAINT,\
                                    pos[0], pos[1], pos[2], axis[0], axis[1],\
                                    axis[2], disableCollisions)
    #Set joint limits
    #if min = max, locked
    #if min < max, constrained to within those
    #if min > max, unconstrained
    joint.setParam(aLab["x"], 0,0)
    joint.setParam(aLab["y"], 0,0)
    joint.setParam(aLab["z"], 1,0) #unconstrained
    joint.setParam(aLab["xr"], 0,0)
    joint.setParam(aLab["yr"], 0,0)
    joint.setParam(aLab["zr"], 0,0)
    
    #make lower to base ball joint
    bge.constraints.createConstraint(lower, base, \
                                    bge.constraints.POINTTOPOINT_CONSTRAINT,\
                                    pos[0], pos[1], pos[2], axis[0], axis[1],\
                                    axis[2], disableCollisions)
    #make upper to platform ball joint
    bge.constraints.createConstraint(upper, platform, \
                                    bge.constraints.POINTTOPOINT_CONSTRAINT,\
                                    pos[0], pos[1], pos[2], axis[0], axis[1],\
                                    axis[2], disableCollisions)
