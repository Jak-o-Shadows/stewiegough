# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:30:27 2015

@author: Jak
"""

import SocketServer

import conn
import sys
sys.path.append("..")
import fk
import math

    
    
class MsgPasser(SocketServer.StreamRequestHandler):
   
    
    def readTestPositions(self):
        with open("positions", "rb") as f:
            data = f.read().split("\n")[:-1]
            self.positions = data
    
    
    def start(self):
        self.started = True
        self.index = 0


    def setup(self):
        self.start()
        self.readTestPositions()
        print('{}:{} connected'.format(*self.client_address))
        

    def handle(self):
        while True:
            data = self.request.recv(4096)
            print "Recived:", data
            
            a = self.positions[self.index]
            self.index += 1
            if self.index >= len(self.positions):
                self.index = 0
            #a = [str(x) for x in a]
            print "Sending:", a
            self.request.sendall(a)

    def finish(self):
        print('{}:{} disconnected'.format(*self.client_address))


def main():
    HOST, PORT = "localhost", 8080

    # Create the server, binding to localhost on port 9999
    server = SocketServer.TCPServer((HOST, PORT), MsgPasser)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
    
    
if __name__ == "__main__":
    main()