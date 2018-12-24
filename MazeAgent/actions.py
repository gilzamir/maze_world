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
def walk(net, speed=1):
	net.act(0.0, speed)

def run(net):
	walk(net, 15.0)

def noop(net):
    pass
	
def walk_in_circle(net, speed=0.5):
	net.act(speed, speed)

def crouch(net):
	net.act(0.0, 0.0, 0.0, True, False, 0.0)

def jump(net):
	net.act(0.0, 0.0, 0.0, False, True, 0.0)
	
def see_around_by_left(net, speed=0.5):
	net.act(0.0, 0.0, 0.0, False, False, speed)
	
def see_around_by_right(net, speed=0.5):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, speed)
	
def see_around_up(net, speed=0.5):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, speed)
	
def see_around_down(net, speed=0.5):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, speed)

def push(net):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, True)
	
def reset_state(net):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, True)

def get_pickup(net):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, True)

def restart(net):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, True)

def pause(net):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, True)

def resume(net):
	net.act(0.0, 0.0, 0.0, False, False, 0.0, 0.0, 0.0, 0.0, False, False, False, True, False, False, True)
