# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:41:12 2015

@author: Jak
"""

import bge
import mathutils

import conn
import queue


#Communication stuff?
th = conn.ThreadHelper(conn.SocketProtocol, conn.message)
connected = False

#Blender stuff
cont = bge.logic.getCurrentController()
obj = cont.owner

def updatePosition(a):
    global obj
    #Update position
    obj.localPosition = a[0:3]
    #Update orientation
    mat_rotz = mathutils.Matrix.Rotation(a[3], 3, 'Z')
    mat_roty = mathutils.Matrix.Rotation(a[4], 3, 'Y')
    mat_rotx = mathutils.Matrix.Rotation(a[5], 3, 'X')
    mat_rot = mat_rotz*mat_roty*mat_rotx
    obj.localOrientation = mat_rot

def connect():
    global th
    global connected
    if not connected:
        print("Connecting")
        th.protocol.connect()
        print("Connected")
        th.sendQueue.put("Connected to Blender")
        th.startThread()
        connected = True
        return True
    else:
        print("Already Connected!")
        return False
        
def disconnect():
    global connected
    if connected:
        global th
        th.protocol.disconnect()
        connected = False
    
    
def recieve():
    global connected
    if connected:
        global th
        try:
            data = th.recvQueue.get(block=False)
            print("Recieved data:")
            a = [float(x) for x in data.split(",")]
            print(a)
            updatePosition(a)
            th.recvQueue.task_done()
        except queue.Empty:
            pass
        
def send():
    global connected
    if connected:
        global th
        st = []
        for i in range(6):
            i += 1
            st .append(str(obj["leg"+str(i)]))
        th.sendQueue.put(",".join(st))
        
