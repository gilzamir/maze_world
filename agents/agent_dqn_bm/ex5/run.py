# -*- coding: utf-8 -*-
import sys
import numpy as np
from bagent import DQNAgent as Agent
from collections import deque
import threading as td
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.transform import rotate
import random
import gym
import io
import MazeWorld
import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
agent = Agent( (10, 10), 7, 2)
agent.load("model4000")
agent.epsilon_decay = 0
agent.epsilon = 0
agent.epsilon_min = 0

def pre_processing(observe):
    observe = np.array(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

LOSS = 0.0

RENDER = False
REFRESH_MODEL_NUM = 10000
N_RANDOM_STEPS = 50000
MAX_EPSODES = 100
NO_OP_STEPS = 2
FRAME_SKIP = 4
NOOP_ACTION = [0,1,6]

env = gym.make('MazeWorld-v0')
log = open('log0.txt', 'w')
for i in range(MAX_EPSODES):
    frame = env.reset()
    if i > 0 and i % 50 == 0:
        log.close()
        log = open('log%d.txt'%(i), 'w')
    
    if RENDER:
        env.render()

    is_done = False

    batch_size = 12
    agent.reset()
    agent.baredom.min = 0.01
    agent.baredom.max = 0.80
    action = 0
    next_state = None
    initial_proprioception = np.array([0, 0, agent.baredom.min, agent.baredom.max, agent.baredom.current_value])

    for _ in range(random.randint(1, NO_OP_STEPS)) :
        action = random.sample(NOOP_ACTION,1)[0]
        frame, _, _, info = env.step(action, FRAME_SKIP)
        initial_proprioception = np.array([ 1 if info['isPickUpNear'] else 0, np.clip(info['nearPickUpValue'], -1, 1), agent.baredom.min, agent.baredom.max, agent.baredom.current_value])

    if not env.useRaycasting:
        frame = pre_processing(frame)

    stack_frame = tuple([frame]*agent.skip_frames)
    initial_state = np.stack(stack_frame, axis=2)
    initial_state = np.reshape([initial_state], (1, agent.skip_frames, env.vresolution, env.hresolution))
    score = 0
    dead = False
    dscore = 0
    while not is_done:
        dead = False
        action = agent.act(initial_state, proprioception=initial_proprioception)
        agent.last_action = action
        baredom_value = agent.baredom.update(agent.step)

        frame, reward, is_done, info = env.step(agent.contextual_actions[action], FRAME_SKIP)
        next_proprioception = np.array([ 1 if info['isPickUpNear'] else 0, np.clip(info['nearPickUpValue'], -1, 1), agent.baredom.min, agent.baredom.max, baredom_value])

        next_frame = frame
        if not env.useRaycasting:
            next_frame = pre_processing(next_frame)
        
        next_state = np.reshape([next_frame], (1, 1, env.vresolution, env.hresolution))
        next_state = np.append(next_state, initial_state[:, :(FRAME_SKIP-1), :, :], axis=1)

        dreward = np.clip(reward, -1.0, 1.0)
        reward = np.clip(reward + 0.0001 * baredom_value, -1.0, 1.0)
  
        score += reward
        dscore += dreward 
    
        end_eps = dead or is_done
        
        if not is_done:
            initial_state = next_state
            initial_proprioception = next_proprioception

        if random.random() <= 0.005:
            print("EPISODE: %d. REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f. TOTAL REWARDS %f" % (
                i, dscore, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step, agent.baredom.min, agent.baredom.max, agent.baredom_freq, score))

        if RENDER:
            env.render()
    
    if (agent.epoch % 1000 == 0):
        agent.save("model%d" % (agent.epoch))

    count_loss = agent.step
    if count_loss == 0:
        count_loss = 1
    now = datetime.datetime.now().time()
    print(">>EPISODE: %d. REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f. TOTAL REWARDS %f" % (
        i, dscore, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step, agent.baredom.min, agent.baredom.max, agent.baredom_freq, score))
    log.write("EPISODE: %d. REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f. TOTAL REWARDS %f\n" % (
        i, dscore, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step, agent.baredom.min, agent.baredom.max, agent.baredom_freq, score))
    log.flush()
    LOSS = 0.0
