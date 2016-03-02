# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:12:16 2015

Set leg lengths to properties

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


#Find leg length

objects = bge.logic.getCurrentScene().objects
platform = objects["platform"]

for i in range(numLegs):
    i+=1
    upper = objects[legPrefix+str(i)+legUpper]
    lower = objects[legPrefix+str(i)+legLower]
    vect = upper.worldPosition - lower.worldPosition
    platform[legPrefix+str(i)] = vect.length



