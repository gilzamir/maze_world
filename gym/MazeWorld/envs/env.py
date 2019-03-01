# -*- coding: utf-8 -*-
import net as ne
import threading as td
import numpy as np
import actions
from PIL import Image
import io
import random
from time import sleep
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import platform

ACTIONS = np.array([actions.noop, actions.walk, actions.walk_in_circle, actions.run, actions.crouch, actions.jump, actions.see_around_by_left, 
			actions.see_around_by_right, actions.see_around_up, actions.see_around_down, actions.reset_state, 
				actions.get_pickup])

ACTION_MEANING = ['WALK', 'WALK AROUND', 'RUN', 'SEE BY LEFT', 'SEE BY RIGHT', 'PICKUP', 'NOOP']

class AleWrapper:
    def __init__(self, env):
        self.env = env
    
    def lives(self):
        return self.env.nlives

class Environment(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, useRayCasting=True):
        self.net = ne.NetCon()        
        self.ale = AleWrapper(self)
        self.game_level = 0
        while not self.net.open():
            self.net.ACT_PORT += 1
            self.net.PERCEPT_PORT += 1
        self.idle = None
        self.initial_energy = 15
        self._action_set = [0, 1, 2, 3, 4, 5, 6]
        self.action_map = {0:1,1:2,2:3,3:6,4:7,5:11,6:0}
    
        self.useRaycasting = useRayCasting
        if useRayCasting:
            self.hresolution = 10
            self.vresolution = 10
            self.maxValue = 6
        else:
            self.hresolution = 84
            self.vresolution = 84
            self.maxValue = 255

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces
        self.observation_space = spaces.Box(low=0, high=self.maxValue, shape=(self.vresolution, self.hresolution, 1), dtype=np.uint8)
        self.last_frame = None

        path = os.environ['MW_PATH']
        
        cmd = None
        systemname = platform.system()
        if systemname=='Windows':
            cmd = 'start %s -screen-fullscreen 0 -screen-height 640 -screen-width 480 --noconfig --input_port %d --output_port %d --game_level %d'%(path, self.net.ACT_PORT, self.net.PERCEPT_PORT, self.game_level)
        elif systemname=='Darwin':
            cmd = 'open -a %s --args -screen-fullscreen 0 -screen-height 640 -screen-width 480 --noconfig --input_port %d --output_port %d --game_level %d &'%(path, self.net.ACT_PORT, self.net.PERCEPT_PORT, self.game_level)
        else:
            cmd = '%s --args -screen-fullscreen 0 -screen-height 640 -screen-width 480 --noconfig --input_port %d --output_port %d  --game_level %d &'%(path, self.net.ACT_PORT, self.net.PERCEPT_PORT, self.game_level)
        os.system(cmd)

    def set_level(self, level):
        self.game_level = level
        actions.set_level(self.net, self.game_level)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


    def _wait_perception(self):
        pair = None
        c = 0
        while pair == None:
            pair = self.net.percept()
            if not pair:
                self.net.update_sensor()
                if c > 100:
                    c = 0
                    print('listening sensor....')
                    sleep(0.2)
            c += 1
        return pair

    def _agent_perception(self, perception=None):
        if (perception):
            p, fr = perception
        else:
            p, fr = self._wait_perception()
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
                frame = np.reshape(m, (self.vresolution, self.hresolution))
            except Exception:
                frame = np.zeros((self.vresolution, self.hresolution))
        perception = [float(t.replace(',', '.')) for t in p.strip().split(';')]
        return (perception,  frame)

    def _get_one_step(self, action):
        ACTIONS[action](self.net)
        if self.idle != None:
            sleep(self.idle)
        return self._agent_perception()

    def reset(self):
        self.initial_energy = 15
        self.prev_perception = None
        actions.restart(self.net)
        sleep(0.2)
        _, frame = self._get_one_step(0)
        self.last_frame = frame
        actions.pause(self.net)
        
        return frame

    def render(self, mode='human', close=False):
        return self.last_frame

    def _prepare_data(self, info, frame, action):
        if not self.useRaycasting:
            frame = Image.open(io.BytesIO(frame))

        pinfos = None
        reward = 0
        get_pickup = False
        if self.prev_perception:
            _, _, _, pinfos = self.prev_perception
            wasPickUpNear = pinfos['isPickUpNear']
            pickUpValue = pinfos['nearPickUpValue']
            if wasPickUpNear and ACTION_MEANING[action] == "PICKUP":
                reward += pickUpValue
                get_pickup = True
        
        lives = info[0]
        self.nlives = lives
        energy = info[1]
        score = info[2]
        done = info[3]
        isPickUpNear = True if (info[4] == 1 and not get_pickup) else False
        nearPickUpValue = info[5]
        orientation = info[6]
        position = np.zeros(3)
        position[0] = info[7]
        position[1] = info[8]
        position[2] = info[9]
        isWithKey = info[10]

        infos = {'lives': lives, 'isWithKey': isWithKey, 'energy': energy, 'isPickUpNear': isPickUpNear, 'nearPickUpValue': nearPickUpValue, 'score': score, 'orientation': orientation, 'position': position}
        self.last_frame = frame
        return frame, reward, done, infos
        
    def close(self):
        actions.close(self.net)

    def step(self, action, frame_skip = 0):
        actions.resume(self.net)
        info = None
        frame = None
        maction = self.action_map[action]
        if frame_skip > 0:
            for _ in range(frame_skip-1):
                ACTIONS[maction](self.net, cmdtype="action ignore frame")
        info, frame = self._get_one_step(maction)
        self.prev_perception = self._prepare_data(info, frame, action)
        actions.pause(self.net)
        return self.prev_perception

    def __del__(self):
        self.net.close()



