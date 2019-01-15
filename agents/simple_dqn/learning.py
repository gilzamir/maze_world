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

agent = Agent( (10, 10), 7)
agent.front2back()
agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/1000000)

def pre_processing(observe):
    observe = np.array(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

LOSS = 0.0

RENDER = False
REFRESH_MODEL_NUM = 10000
N_RANDOM_STEPS = 50000
MAX_EPSODES = 100000000
NO_OP_STEPS = 30
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
    action = 0
    next_state = None

    for _ in range(random.randint(1, NO_OP_STEPS)) :
        action = random.sample(NOOP_ACTION,1)[0]
        frame, _, _, info = env.step(action, FRAME_SKIP)

    if not env.useRaycasting:
        frame = pre_processing(frame)

    stack_frame = tuple([frame]*agent.skip_frames)
    initial_state = np.stack(stack_frame, axis=2)
    initial_state = np.reshape([initial_state], (1, agent.skip_frames, env.vresolution, env.hresolution))
    score = 0
    dead = False

    while not is_done:
        dead = False
        if agent.global_step >= N_RANDOM_STEPS:
            action = agent.act(initial_state)
        else:
            action = agent.act(initial_state, True)

        frame, reward, is_done, info = env.step(agent.contextual_actions[action], FRAME_SKIP)

        next_frame = frame
        if not env.useRaycasting:
            next_frame = pre_processing(next_frame)
        
        next_state = np.reshape([next_frame], (1, 1, env.vresolution, env.hresolution))
        next_state = np.append(next_state, initial_state[:, :(FRAME_SKIP-1), :, :], axis=1)
        
        reward = np.clip(reward, -1.0, 1.0)

        score += reward
    
        end_eps = dead or is_done
        
        agent.remember(initial_state, action, reward, next_state, end_eps)
        if (agent.global_step >= N_RANDOM_STEPS):
            LOSS += agent.replay(batch_size)
            if agent.global_step % REFRESH_MODEL_NUM == 0:
                agent.front2back()

        if not is_done:
            initial_state = next_state

        if random.random() <= 0.005:
            print("EPISODE: %d. SUM OF REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f." % (
                i, score, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step))

        if RENDER:
            env.render()
    
    if (agent.epoch % 1000 == 0):
        agent.save("model%d" % (agent.epoch))

    count_loss = agent.step
    if count_loss == 0:
        count_loss = 1
    now = datetime.datetime.now().time()
    print("EPISODE IS OVER. EPISODE: %d. SUM OF REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Time: %s" % (
        i, score, agent.epsilon, agent.step, agent.global_step, LOSS/count_loss, now))
    log.write("EPISODE IS OVER. EPISODE: %d. SUM OF REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Time: %s\n" % (
        i, score, agent.epsilon, agent.step, agent.global_step, LOSS/count_loss, now))
    log.flush()
    LOSS = 0.0
