import net as ne
import threading as td
import numpy as np
import actions
from PIL import Image
import io
import random
from time import sleep

ACTIONS = np.array([actions.noop, actions.walk, actions.walk_in_circle, actions.run, actions.crouch, actions.jump, actions.see_around_by_left, 
			actions.see_around_by_right, actions.see_around_up, actions.see_around_down, actions.reset_state, 
				actions.get_pickup])
PROPRIOCEPTION = 0
SENSORS = 1

class Environment:
    def __init__(self, idle=None, useRaycasting = True):
        self.net = ne.NetCon()
        self.net.open()
        self.idle = idle
        self.initial_energy = 15
        self.useRaycasting = useRaycasting
        self.hresolution = 10
        self.vresolution = 10
        self.maxValue = 6

    def render(self):
        pass

    def agent_perception(self):
        p, fr = self.net.percept()
        p = str(p, 'utf-8')
        frame = fr
        if self.useRaycasting:
            try:
                m = np.zeros((self.vresolution, self.hresolution))
                f = str(fr, 'utf-8')
                lines = f.strip().split(";")
            
                i = 0
                for line in lines:
                    values = line.strip().split(",")
                    j = 0
                    for value in values:
                        if len(value) > 0:
                            m[i][j] = int(value)
                            j += 1
                    i += 1
                frame = m
            except Exception:
                frame = np.zeros((self.vresolution, self.hresolution))

        #print(p)
        perception = [float(t) for t in p.strip().split(';')]
        return (perception,  frame)

    def get_one_step(self, action):
        ACTIONS[action](self.net)
        if self.idle != None:
            sleep(self.idle)
        return self.agent_perception()

    def reset(self):
        actions.resume(self.net)
        actions.restart(self.net)
        info, frame = self.get_one_step(0)
        return self.prepare_data(info, frame)

    def prepare_data(self, info, frame):
        if not self.useRaycasting:
            frame = Image.open(io.BytesIO(frame))
        #print(info)
        lives = info[0]
        energy = info[1]
        #score = info[2]
        done = info[-3]
        isPickUpNear = True if info[-2] == 0 else False
        nearPickUpValue = info[-1]

        reward = energy - self.initial_energy
        self.initial_energy = energy

        infos = {'lives': lives, 'energy': energy, 'isPickUpNear': isPickUpNear, 'nearPickUpValue': nearPickUpValue}

        return frame, reward, done, infos
        

    def step(self, action, frame_skip = 4):
        actions.resume(self.net)
        info = None
        frame = None
        if frame_skip > 0:
            for _ in range(frame_skip-1):
                ACTIONS[action](self.net)
        info, frame = self.get_one_step(action)
        actions.pause(self.net)
        return self.prepare_data(info, frame)
    
    def __del__(self):
        self.net.close()


