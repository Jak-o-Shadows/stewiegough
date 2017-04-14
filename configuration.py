# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 14:29:07 2015

@author: Jak
"""
import math

#define coord system origin as the centre of the bottom plate
#Find base plate attachment locations
bAngles = [-5, 5, 115, 125, 235, 245]
#bAngles = range(0, 360, 360/6)
bAngles = [math.radians(x) for x in bAngles]
bR = 50
bR = 100
bPos = [[bR*math.cos(theta), bR*math.sin(theta), 0] for theta in bAngles]

#bPos = [[55, -20, 0], [55, 20, 0], [15, 40, 0], [-55, 20, 0], [-55, -20, 0], [-15, -40, 0]]

#Platform attachment locations
#pAngles = [-30, 30, 90, 150, 210, 270]
#pAngles = range(15, 360+15, 360/6)
pAngles = [305, 55, 65, 175, 185, 295]

pAngles = [math.radians(x) for x in pAngles]
pR = 32
pR = bR#*0.75
pPos = [[pR*math.cos(theta), pR*math.sin(theta), 0] for theta in pAngles]



height = 160

legMin = [50]*6
legMax = [100]*6

#Base UV joint limits
A = [math.pi/4]*6
#Platform ball joint limits
B = [math.pi/2]*6

#import pprint
#pprint.pprint(bPos)
#pprint.pprint(pPos)