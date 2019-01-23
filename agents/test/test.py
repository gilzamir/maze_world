# -*- coding: utf-8 -*-
import random
import MazeWorld
import gym


MAX = 100000
i = 0
env = gym.make('MazeWorld-v0')
    
while i < MAX:
    is_done = False
    frame = env.reset()[0]
    while not is_done:
        action = random.sample([0, 6], 1)[0]
        #action = 1
        #print("Input action: ")
        #action = int(input())
        reward = None
        if action == -1:
            frame = env.reset()
        else:
            frame, reward, is_done, _ = env.step(action, 4)
        i += 1
    if i % 1000:
        print(i)