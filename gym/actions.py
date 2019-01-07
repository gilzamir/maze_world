# -*- coding: utf-8 -*-
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
# rf: release the flag.
# gf: get the flag.
# ws: apply walk speed. 
# rs: restart
##############################################################################################
def walk(net, speed=1, cmdtype="action"):
	net.act(0.0, speed, cmdtype=cmdtype)

def run(net, cmdtype="action"):
	walk(net, 15, cmdtype=cmdtype)

def noop(net, cmdtype="action"):
    net.update_sensor()

def walk_in_circle(net, speed=1, cmdtype="action"):
	net.act(speed, speed, cmdtype=cmdtype)

def crouch(net, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, True, False, 0.0, cmdtype=cmdtype)

def jump(net, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, True, 0.0, cmdtype=cmdtype)
	
def see_around_by_left(net, speed=1, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, False, speed, cmdtype=cmdtype)
	
def see_around_by_right(net, speed=1, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, speed, cmdtype=cmdtype)
	
def see_around_up(net, speed=1, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, speed, cmdtype=cmdtype)
	
def see_around_down(net, speed=1, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, speed, cmdtype=cmdtype)

def push(net, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, True, cmdtype=cmdtype)
	
def reset_state(net, cmdtype="control"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, True, cmdtype=cmdtype)

def get_pickup(net, cmdtype="action"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, True, cmdtype=cmdtype)

def restart(net, cmdtype="control"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, True, cmdtype=cmdtype)

def pause(net, cmdtype="control"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, True, cmdtype=cmdtype)

def resume(net, cmdtype="control"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, False, True, cmdtype=cmdtype)

def close(net, cmdtype="control"):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, False, False, True, cmdtype=cmdtype)
