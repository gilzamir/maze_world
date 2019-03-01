# -*- coding: utf-8 -*-
import socket
from threading import Thread
import time
import sys


class NetCon:
    def __init__(self, host="127.0.0.1", bs=5000):
        self.ACT_PORT = 8870
        self.PERCEPT_PORT = 8890
        self.HOST = host
        self.PERCEPT_BUFFER_SIZE = bs
        self.UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def percept(self):
        return self.receive_data()

    ############################################################################################
    # act parameters order
    # fx: horizontal direction of movement.
    # fy: vertical direction of movement.
    # speed: velocity of movement, let it on 0.5 speed.
    # crouch: let it crouch (agachar in portuguese).
    # jump: let it jump (pular in portuguese).
    # l (left): left rotation of the head.
    # r (right): right rotation of the head.
    # up: vertical to up rotation of the head.
    # d (down): vertical to down rotation of the head.
    # ps: push
    # rf: reset state.
    # gf: get the pickup.
    # ws: apply walk speed.
    # rs: restart
    ##############################################################################################
    def act(self, fx, fy, speed=0.5, crouch=False, jump=False, l=0.0, r=0.0, u=0.0, d=0.0, ps=False, rf=False, gf=False, ws=True, rs=False, pause=False, resume=False, closefn=False, cmdtype="action"):
        self.send_command(fx, fy, speed, crouch, jump,
                     l, r, u, d, ps, rf, gf, ws, rs, pause, resume, closefn, cmdtype)

    def close(self):
        self.UDP.close()

    def open(self):
        return self.open_receive()


    def open_receive(self):
        try:
            server_address = (self.HOST, self.PERCEPT_PORT)
            print('starting up on %s port %s' % server_address)
            self.UDP.settimeout(0.1)
            self.UDP.bind(server_address)
            return True
        except:
            return False

    def receive_data(self):
        # Bind the socket to the port
        return self.recvall()

    def recvall(self):
        data = bytearray(self.PERCEPT_BUFFER_SIZE)
        try:
            data, _ = self.UDP.recvfrom(self.PERCEPT_BUFFER_SIZE)
            return (data[0:70], data[70::])
        except:
            #e = sys.exc_info()[1]
            #print("Error: %s\n" % e)
            #raise
            return None

    ############################################################################################
    # Send commands to remote controller of the player.
    # HOST: remote host
    # PORT: communication port
    # socket: socket object to communication
    # fx: horizontal direction of movement.
    # fy: vertical direction of movement.
    # speed: velocity of movement, let it on 0.5 speed.
    # crouch: let it crouch (agachar in portuguese).
    # jump: let it jump (pular in portuguese).
    # l (left): left rotation of the head.
    # r (right): right rotation of the head.
    # up: vertical to up rotation of the head.
    # d (down): vertical to down rotation of the head.
    # ps: push
    # rf: reset state.
    # gf: get the pickup.
    # ws: apply walk speed.
    ##############################################################################################
    def send_command(self, fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs=False, pause=False, resume=False, closefn=False, cmdtype="action"):
        command = "%s;%f;%f;%f;%r;%r;%f;%f;%f;%f;%r;%r;%r;%r;%r;%r;%r;%r" % (
            cmdtype,fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs, pause, resume, closefn)
        dest = (self.HOST, self.ACT_PORT)
        self.UDP.sendto(command.encode(encoding="utf-8"), dest)
    
    def update_sensor(self):
        dest = (self.HOST, self.ACT_PORT)
        self.UDP.sendto("update_sensor".encode(encoding="utf-8"), dest)


