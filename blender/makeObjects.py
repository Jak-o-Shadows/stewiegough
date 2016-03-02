# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 11:27:05 2015

@author: Jak
"""

import math
import sys
import os
sys.path.append(".." + os.sep)
print(os.getcwd())

import bpy

#Load stewart configuration
from configuration import *


#default units is mm
thickness = height/10

def makeRigidBody(obj):
    """Changes the object physics setting to be rigid body
    Uses operator - bad?
    """
    obj.game.physics_type = "RIGID_BODY"
    obj.game.use_ghost = True
    obj.game.use_sleep = True # disable sleeping

def consistentNormals(obj):
    """Make normals consistent & all pointing out"""
    bpy.context.scene.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()


def makeHexagon(coords, thickness, z, name):
    #create vertices
    vertsTop = []
    vertsBottom = []
    for c in coords:
        vertsBottom.append(c)
        vertsTop.append((c[0], c[1], c[2] + thickness))
    verts = vertsTop + vertsBottom + [(0,0,0), (0,0,thickness)]
    #create faces
    faces = []
    #top and bottom faces
    for i in range(6):
        if i != 5:
            #Top
            a = i
            b = i+1
            c = 13 #last one - center point
            faces.append((a, b, c))
           #bottom
            a = i+6
            b = i+1 + 6
            c = 12
            faces.append((a, b, c))
        else:
            faces.append((5, 0, 13))
            faces.append((11, 6, 12))

    #side faces
    
    for i in range(6):
        if i!=5:
            a = i
            b = i+6
            d = i+1
            c = i+1 + 6
            faces.append((a, b, c, d))
        else:
            faces.append((5, 5+6, 0+6, 0))

    
    
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    obj.location = [0,0,z]
    bpy.context.scene.objects.link(obj)

    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    
    consistentNormals(obj)
    
    return obj
    
def makeRecPrism(faceCoords, length, origin, name):
    verts = [tuple(c) for c in faceCoords]
    verts += [(c[0], c[1], c[2]+length) for c in faceCoords]
    #Faces
    faces = []
    #End-caps
    faces.append((0,1,2,3))
    faces.append((4,5,6,7))
    #sides
    for i in range(1,4):
        a = i
        b = i-1
        c = i-1+4
        d = i+4
        faces.append((a, b, c, d))
    faces.append((3,0,4,7))
    #print(faces)
    #print(verts)
        
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    obj.location = origin
    bpy.context.scene.objects.link(obj)
    
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    
    consistentNormals(obj)
    return obj
    
def makeLegs(bPos, pPos, height, zOffset, lengthRatio, legRatio):
    for i in range(6):
        b = bPos[i]
        p = pPos[i]
        
        length = math.sqrt((b[0]-p[0])**2 + (b[1]-p[1])**2 + (b[2]-p[2] + height)**2)
        lengthU = lengthRatio*length/legRatio
        lengthL = lengthRatio*length*legRatio
        lowerOrigin = [o for o in b]
        lowerOrigin[2] += zOffset
        upperOrigin = [o for o in p]
        upperOrigin[2] += zOffset+height
        
        #Size of leg
        w = zOffset
        d = w/2
        faceUpper = [(-w/2, 0, 0), (-w/2, d, 0), (w/2, d, 0), (w/2, 0, 0)]
        faceLower = [(w/2, 0, 0), (w/2, -d, 0), (-w/2, -d, 0), (-w/2, 0, 0)]
        
        #Calculate rotations
        dx = b[0] - p[0]
        dy = b[1] - p[1]
        dz = b[2] - p[2] + height
        
        roll = -math.acos(dy/length) + math.pi/2
        pitch = math.acos(dx/length) - math.pi/2
        yaw = 0

        legLower= makeRecPrism(faceLower, lengthL, lowerOrigin, "leg"+str(i+1)+"Lower")
        legLower.rotation_mode = "XYZ"
        legLower.rotation_euler = (roll, pitch, yaw)
        makeRigidBody(legLower)
        
        legUpper= makeRecPrism(faceUpper, -lengthU, upperOrigin,"leg"+str(i+1)+"Upper")
        legUpper.rotation_mode = "XYZ"
        legUpper.rotation_euler = (roll, pitch, yaw)
        makeRigidBody(legUpper)
        

makeHexagon(bPos, thickness, 0, "base")
platform= makeHexagon(pPos, thickness, height+thickness, "platformToMerge")
makeRigidBody(platform)
makeLegs(bPos, pPos, height, thickness, 3/5, 1)


        
        












