from net import act, percept
import net
import threading as td
import numpy as np


PROPRIOCEPTION = 0
SENSORS = 1

def agent_perception():
	(size, data) = percept()
	if size <= 100:
		p = str(data, 'utf-8')
		perception = [float(t) for t in p.strip().split(';')]
		return (PROPRIOCEPTION, size, perception)
	else:
		return (SENSORS, size,  data)

def agent_act(state):
	if (state[0] == PROPRIOCEPTION):
		print("I see myself: %s"%(state[2]))
		f = np.random.normal(1, 5, 7)
		act(f[0], f[1], f[2], bool(np.random.binomial(1, 0.5)), bool(np.random.binomial(1, 0.5)),
		f[3], f[4],f[5], f[6], bool(np.random.binomial(1, 0.5)))
	else:
		#img = open('frame_%s_.jpg'%(count), "wb")
		#count = count + 1
		#img.write(bytes(state[2]))
		#img.close()
		print("I see the world!!!")

net.open()
for i in range(1,100000):
	agent_act(agent_perception())
net.close()
