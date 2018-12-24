import socket
from threading import Thread
import time
import sys


class NetCon:
    def __init__(self, action_port=8881, perception_port=8888, host="127.0.0.1", bs=4000):
        self.ACT_PORT = action_port
        self.PERCEPT_PORT = perception_port
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
    def act(self, fx, fy, speed=0.5, crouch=False, jump=False, l=0.0, r=0.0, u=0.0, d=0.0, ps=False, rf=False, gf=False, ws=True, rs=False, pause=False, resume=False):
        self.send_command(fx, fy, speed, crouch, jump,
                     l, r, u, d, ps, rf, gf, ws, rs, pause, resume)

    def close(self):
        self.UDP.close()


    def open(self):
        self.open_receive()


    def open_receive(self):
        server_address = (self.HOST, self.PERCEPT_PORT)
        print('starting up on %s port %s' % server_address)
        self.UDP.bind(server_address)


    def receive_data(self):
        # Bind the socket to the port
        return self.recvall()


    def recvall(self):
        data = bytearray(self.PERCEPT_BUFFER_SIZE)
        try:
            data, addr = self.UDP.recvfrom(self.PERCEPT_BUFFER_SIZE)
            return (data[0:50], data[50::])
        except:
            e = sys.exc_info()[1]
            print("Error: %s\n" % e)
            raise

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
    def send_command(self, fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs=False, pause=False, resume=False):
        command = "%f;%f;%f;%r;%r;%f;%f;%f;%f;%r;%r;%r;%r;%r;%r;%r" % (
            fx, fy, speed, crouch, jump, l, r, u, d, ps, rf, gf, ws, rs, pause, resume)
        # print(command)
        dest = (self.HOST, self.ACT_PORT)
        #print('sending to %s port %s' % dest)
        self.UDP.sendto(command.encode(), dest)
