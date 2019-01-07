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
import MazeWorld

agent = Agent( (10, 10), 7)
agent.front2back()
agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/500000)

def pre_processing(observe):
    observe = np.array(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

LOSS = 0.0

def back2front(agent, loss):
    global LOSS
    agent.loss += loss
    agent.count_loss += 1
    LOSS += loss
    agent.back2front()


RENDER = False
REFRESH_MODEL_NUM = 10000
N_RANDOM_STEPS = 5000
MAX_EPSODES = 100000000
NO_OP_STEPS = 30
FRAME_SKIP = 10
NOOP_ACTION = 6

env = gym.make('MazeWorld-v0')

for i in range(MAX_EPSODES):
    frame = env.reset()
    if RENDER:
        env.render()

    is_done = False

    batch_size = 12
    agent.reset()
    action = 0
    next_state = None

    for _ in range(NO_OP_STEPS):
        frame = env.step(NOOP_ACTION, FRAME_SKIP)[0]

    if not env.useRaycasting:
        frame = pre_processing(frame)

    stack_frame = tuple([frame]*agent.skip_frames)
    initial_state = np.stack(stack_frame, axis=2)
    initial_state = np.reshape([initial_state], (1, env.vresolution, env.hresolution, agent.skip_frames))
    score = 0
    dead = False
    freq = np.zeros(agent.action_size)
    while not is_done:
        dead = False
        if agent.global_step >= N_RANDOM_STEPS:
            action = agent.act(initial_state)
        else:
            action = agent.act(initial_state, True)
        freq[action] += 1
        frame, reward, is_done, info = env.step(agent.contextual_actions[action], FRAME_SKIP)

        next_frame = frame
        if not env.useRaycasting:
            next_frame = pre_processing(next_frame)
        next_state = np.reshape([next_frame], (1, env.vresolution, env.hresolution, 1))
        next_state = np.append(next_state, initial_state[:, :, :, :(FRAME_SKIP-1)], axis=3)
        score += reward
        reward = np.clip(reward, -1.0, 1.0)
    
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
    
    if (agent.epoch % 100 == 0):
        agent.save("model%d" % (agent.epoch))

    count_loss = agent.step
    if count_loss == 0:
        count_loss = 1

    print("EPISODE IS OVER. EPISODE: %d. SUM OF REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f." % (
        i, score, agent.epsilon, agent.step, agent.global_step, LOSS/count_loss))
    LOSS = 0.0
