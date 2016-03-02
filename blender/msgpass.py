# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:11:15 2013

@author: Jak
"""

import conn

import Queue as queue #Python 2.x uses Queue, Python 3.x uses queue

class SerialTCPBridge():
    """Sends serial data to TCP & vice versa"""
    def __init__(self, TCPHOST, TCPPORT, COMPORT, BAUDRATE):
        self.host = TCPHOST
        self.port = TCPPORT
        
        #Setup Serial Connection
        self.serial = conn.ThreadHelper(conn.SerialProtocol, conn.message)
        self.serial.protocol.port = COMPORT
        self.serial.protocol.baudrate = BAUDRATE
        #Setup TCP connection
        self.tcp = conn.ThreadHelper(conn.SocketProtocol, conn.message)
        self.tcp.protocol.address = (TCPHOST, TCPPORT)
        
    def connectSerial(self):
        self.serial.protocol.connect()
        self.serial.startThread()
    
    def connectTCP(self):
        self.tcp.protocol.connect()
        self.tcp.startThread()
    
    def connect(self):
        self.connectSerial()
        self.connectTCP()
        
    def serve(self):
        #DO ERROR CHECKING HERE! ie. have we actually connected
        
        #check if any data has arrived on serial & send it to tcp
        try:
            data = self.serial.recvQueue.get(block=False)
            self.tcp.sendQueue.put(data)
            self.serial.recvQueue.task_done()
        except queue.Empty:
            pass
        #check if any data has arrived on tcp & send it to serial
        try:
            data = self.tcp.recvQueue.get_nowait()
            self.serial.sendQueue.put(data)
            self.tcp.recvQueue.task_done()
        except queue.Empty:
            pass


def main():
    HOST, PORT = "localhost", 8080
    COM, BAUD = "COM3", 9600
    s = SerialTCPBridge(HOST, PORT, COM, BAUD)
    s.connect()
    s.serve()

    
    
if __name__ == "__main__":
    main()