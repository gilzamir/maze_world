import sys 
sys.path.append('..')
import env as e
import numpy as np
from bagent import DQNAgent as Agent
from collections import deque
import threading as td
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.transform import rotate
import random

agent = Agent((84, 84), 6)
agent._build_model()
agent.front2back()
agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/1000000)


def pre_processing(observe):
    observe = np.array(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

#TRANSFORM_OBS_PROB = 0.00
'''
def pre_processing(observe):
    observe = np.array(observe)
    if (np.random.rand() <= TRANSFORM_OBS_PROB):
        if (np.random.rand()<=0.5):
            observe = rotate(observe, -90)
        else:
            observe = rotate(observe, 90)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe
'''

LOSS = 0.0


def back2front(agent, loss):
    global LOSS
    agent.loss += loss
    agent.count_loss += 1
    LOSS += loss
    agent.back2front()


RENDER = False
REFRESH_MODEL_NUM = 10000
N_RANDOM_STEPS = 50000
MAX_EPSODES = 100000000
NO_OP_STEPS = 30

env = e.Environment()

for i in range(MAX_EPSODES):
    frame = env.reset()
    if RENDER:
        env.render()

    is_done = False

    batch_size = 12
    batch_size3 = 3*12
    score, start_life = 0, 5
    agent.reset()
    action = 0
    next_state = None

    # this is one of DeepMind's idea.
    # just do nothing at the start of episode to avoid sub-optimal
    for _ in range(NO_OP_STEPS):
        frame = env.step(1)[0]

    frame = pre_processing(frame)
    stack_frame = tuple([frame]*agent.skip_frames)
    initial_state = np.stack(stack_frame, axis=2)
    initial_state = np.reshape([initial_state], (1, 84, 84, agent.skip_frames))

    dead = False
    while not is_done:
        dead = False
        if agent.global_step >= N_RANDOM_STEPS:
            action = agent.act(initial_state)
        else:
            action = agent.act(initial_state, True)
        frame, reward, is_done, info = env.step(agent.contextual_actions[action])

        next_frame = pre_processing(frame)
        next_state = np.reshape([next_frame], (1, 84, 84, 1))
        next_state = np.append(next_state, initial_state[:, :, :, :3], axis=3)
        
        score += reward
        
        if start_life > info['lives']:
            reward = -1
            dead = True
            start_life = info['lives']

        reward = np.clip(reward, -1.0, 1.0)
        print(reward)
        end_eps = dead or is_done
        

        if (agent.global_step >= N_RANDOM_STEPS and (not agent.replay_is_running)):
            replay_is_running = True
            LOSS += agent.replay(batch_size)
            if agent.global_step % REFRESH_MODEL_NUM == 0:
                agent.back2front()

        if not is_done:
            initial_state = next_state

        if RENDER:
            env.render()
        if (agent.epoch % 1000 == 0):
            agent.save("model%d" % (agent.epoch))

    count_loss = agent.step
    if count_loss == 0:
        count_loss = 1

    print("SCORE ON EPISODE %d IS %d. EPSILON IS %f. STEPS IS %d. GSTEPS IS %d. LOSS IS %f" % (
        i, score, agent.epsilon, agent.step, agent.global_step, LOSS/count_loss))
    LOSS = 0.0
