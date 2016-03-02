# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:40:15 2016

@author: Jak
"""


import sys
sys.path.append("..//blender/")
sys.path.append("../")

import conn

import rotary



def calibrate():
    port = "COM4"
    baud = 9600
    
    t = conn.ThreadHelper(conn.SerialProtocol, conn.message)    
    t.protocol.port = port
    t.protocol.baud = baud
    t.protocol.connect()
    
    
    

    t.startThread()
    
    
    while 1:
        data = t.recvQueue.get().rstrip("\r").split(";")
        print data

def  main():
    port = "COM4"
    baud = 9600
    
    
#    tcp = conn.ThreadHelper(conn.SocketServer, lambda x: x)
#    tcp.protocol.address = ("localhost", 8080)
#    tcp.protocol.connect()
#    tcp.startThread()
    
    t = conn.ThreadHelper(conn.SerialProtocol, conn.message)    
    t.protocol.port = port
    t.protocol.baud = baud
    t.protocol.connect()
    
    
    

    t.startThread()
        
    
    
    
    
    zeroPos = [435, 652, 486, 443, 504, 500]
    ninety = [900, 325, 153, 803, 130, 217]
    
    
    zeroPos = [567, 664, 483, 499, 438, 510]
    ninety = [870, 339, 165, 787, 110, 181]  
    
    
    while 1:
        data = t.recvQueue.get().rstrip("\r").split(";")
        data = [int(x) for x in data]        
        print data
        angle = [90 - 90*(data[i] - ninety[i])/float(zeroPos[i] - ninety[i]) for i in xrange(6)]        
        print angle
        c = rotary.ConfigBased()
        pos =  c.fk(angle)
        pos[0] = pos[0]/1000.0
        pos[1] = pos[1]/1000.0
        pos[2] = pos[2]/1000.0
        print "%1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f \n" % tuple([x for x in pos])
        
        
        pos = [str(x) for x in pos]
        
        
        #print ",".join(pos)
        #tcp.sendQueue.put(",".join(pos))
        
        
        t.recvQueue.task_done()
        


        print
    
    
    
    
    
    
    pass




if __name__ == "__main__":
    #calibrate()
    main()