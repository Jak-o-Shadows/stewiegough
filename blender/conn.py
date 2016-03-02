# -*- coding: utf-8 -*-
"""
Created on Tue Jul 09 17:15:30 2013

@author: Jak

Handles the connection and queues and threading

"""
from __future__ import print_function #running this same code in python 2.7.3
import socket
import threading
import time
#import io
import sys

class Protocol():
    """Implements the actual send/recieve interface"""
    def __init__(self):
        pass
    

def message(data):
    """Converts some data into the stuff that gets sent over the connection
    Can replace me! Use if changing line ending for example.
    Copy the bytes(str(data)+ etc), etc though
    """
    print("Message is:",data)
    try:
        return bytes(str(data) + ";", "utf-8")
    except TypeError:
        return bytes(str(data) + ";")
    
    
try:
    if sys.version_info.major >2:
        import queue   
        class BluetoothProtocol():
            """Connects to a bluetooth to serial adaptor"""
            def __init__(self):
                self.MAC = "00:00:00:00:00:00"
                self.port = 3
            
            def connect(self):
                #self.con = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                self.con = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                self.con.connect((self.MAC, self.port))
                
            def disconnect(self):        
                raise NotImplementedError("Disconnect not implemented for bluetooth socket" )
                
            def send(self, msg):
                self.con.send(msg)
                
            def read(self):
                pass
    else:
        import Queue as queue
        import bluetooth
        class BluetoothProtocol():
            """Connects to a bluetooth to serial adaptor using pybluez"""
            def __init__(self):
                self.MAC = "00:00:00:00:00:00"
                self.port = 3
            
            def connect(self):
                #self.con = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
                self.con = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.con.connect((self.MAC, self.port))
                
            def disconnect(self):
                raise NotImplementedError("Disconnect not implemented for pyBluez bluetooth socket" )
                
            def send(self, msg):
                self.con.send(msg)
                
            def read(self):
                pass
except ImportError:
    pass
            

try:
    import serial
    class SerialProtocol(Protocol):
        """Serial protocol, defaults to COM3 and 9600br"""
        def __init__(self):
            self.port = "COM3"
            self.baudrate = 9600
            
            self.eol = "\n"
            
            self.buffer = []
        
        def connect(self):
            self.ser = serial.Serial(port=self.port, baudrate=self.baudrate)
            #self.sio = io.TextIOWrapper(io.BufferedRWPair(self.ser, self.ser, newline=self.eol))
            
        def disconnect(self):
            self.ser.close()
        
        def send(self, msg):
            #self.sio.write(unicode(msg))
            self.ser.write(unicode(msg))
        
        def readBit(self):
            data = self.ser.read(1)
            
            if data != self.eol:
                self.buffer.append(data)
                return True
            else:
                return False
                
        def read(self):
            
            keepGoing = True
            while keepGoing:
                status = self.readBit()
                if not status:
                    data = self.buffer
                    self.buffer = []
                    return "".join(data)
            
            
except ImportError:
    print("Serial Unavailable")
    pass

class SocketProtocol(Protocol):
    """Send and recieve messages over an tcp/ip socket
    Change self.address = (adddr, port). Defaults to ('localhost', 8080)
    """
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = ("localhost", 8080)
              
        
    def connect(self):
        self.sock.connect(self.address)
        
    def disconnect(self):
        self.sock.close()
        
    def send(self, msg):
        self.sock.send(msg)
        
    def read(self):
        l = self.sock.recv(4096)
        return l

class SocketServer(Protocol):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = ("localhost", 8080)
        
    def connect(self):
        """
        Serve as a socket server
        """
        self.sock.bind(self.address)
        self.sock.listen(1)
        
        self.con, self.addr = self.sock.accept()
        
    def send(self, msg):
        self.con.sendall(msg)
        
    def read(self):
        pass
        #print("read not implemented")
        
        
class ThreadHelper():
    """Manages the thread.
    Use ThreadHelper.q.put(item) to put stuff onto the queue for the thread
    """
    def __init__(self, protocol, message):
        self.sendQueue = queue.Queue()
        self.recvQueue = queue.Queue()
        self.protocol = protocol()
        self.msg = message
        
    def startThread(self):
        t = threading.Thread(target=self.sendThread)
        t.daemon = True
        t.start()
        r = threading.Thread(target=self.recvThread)
        r.daemon = True
        r.start()
    
    def sendThread(self):
        keepGoing = True
        while keepGoing:
            item = self.sendQueue.get()
            self.protocol.send(self.msg(item))
            time.sleep(0.01)
            self.sendQueue.task_done()
            
    def recvThread(self):
        keepGoing = True
        while keepGoing:
            data = self.protocol.read()
            try:
                #python 3
                data = str(data, 'utf-8')
            except:
                #python 2.7x
                data = str(data)
            self.recvQueue.put(data)
        

