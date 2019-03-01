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

agent = Agent( (10, 10), 7, 13)
agent.performer.front2back()
agent.goal_searcher.front2back()
agent.epsilon_decay = ((agent.epsilon - agent.epsilon_min)/2000000)
agent.epsilon_min = 0.01

agent.gepsilon_decay = ((agent.gepsilon - agent.gepsilon_min)/1000)
agent.gepsilon_min = 0.10

def pre_processing(observe):
    observe = np.array(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

LOSS = 0.0
LOSS2 = 0.0

RENDER = False

REFRESH_MODEL_NUM = 20000
N_RANDOM_STEPS = 50000

REFRESH_MODEL_NUM2 = 10
N_RANDOM_STEPS2 = 100

MAX_EPSODES = 100000000
NO_OP_STEPS = 30
FRAME_SKIP = 4
NOOP_ACTION = [0,1,6]

env = gym.make('MazeWorld-v0')
log = open('log0.txt', 'w')
logg = open('logg.txt', 'w')
goal_selected = 0
count_eps = 0
global_reward = 0.0
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
    initial_proprioception = np.array([0, 0, agent.baredom.min, agent.baredom.max, agent.baredom.current_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for _ in range(random.randint(1, NO_OP_STEPS)):
        action = random.sample(NOOP_ACTION,1)[0]
        frame, _, _, info = env.step(action, FRAME_SKIP)
        position = info['position']
        if not type(agent.prev_position) == list:
            agent.prev_position = position
        orientation = info['orientation']
        initial_proprioception = [ 1 if info['isPickUpNear'] else 0, np.clip(info['nearPickUpValue'], -1, 1), agent.baredom.min,
                                     agent.baredom.max, agent.baredom.current_value,  position[0], position[1],
                                        position[2], orientation, 0.0, 0.0, 0.0, 0.0]
        initial_proprioception = np.array(initial_proprioception)

    if not env.useRaycasting:
        frame = pre_processing(frame)

    stack_frame = tuple([frame]*agent.skip_frames)
    initial_state = np.stack(stack_frame, axis=2)
    initial_state = np.reshape([initial_state], (1, agent.skip_frames, env.vresolution, env.hresolution))
    dead = False
    sum_dreward = 0
    sum_reward = 0
    sum_reward2 = 0

    if agent.gglobal_step == 0:
        if agent.gglobal_step >= N_RANDOM_STEPS2:
            goal_selected = agent.goal_select(initial_state, proprioception=initial_proprioception[0:5])
        else:
            goal_selected = agent.goal_select(initial_state, True, proprioception=initial_proprioception[0:5])
            
        print('GOAL EPISODE %d. GOAL SELECTED: %d.'%(agent.gglobal_step - 1, goal_selected))

    agent.energy = 0
    while not is_done:
        dead = False
        if agent.global_step >= N_RANDOM_STEPS:
            action = agent.act(initial_state, proprioception=initial_proprioception)
        else:
            action = agent.act(initial_state, True, proprioception=initial_proprioception)
        agent.last_action = action
        agent.baredom_value = agent.baredom.update(agent.step)

        frame, last_reward, is_done, info = env.step(agent.contextual_actions[action], FRAME_SKIP)
        orientation = info['orientation']
        position = info['position']
        score = info['score']
        agent.sum_of_rewards += last_reward
        agent.position = position
        agent.isWithKey = info['isWithKey']
        agent.energy = info['energy']
        agent.last_reward = last_reward
        
        next_proprioception = [ 1 if info['isPickUpNear'] else 0, np.clip(info['nearPickUpValue'], -1, 1), 
                                agent.baredom.min, agent.baredom.max, agent.baredom_value] + list(position) + [orientation] + agent.last_goals_values
        
        next_proprioception = np.array(next_proprioception)

        next_frame = frame
        if not env.useRaycasting:
            next_frame = pre_processing(next_frame)
        
        next_state = np.reshape([next_frame], (1, 1, env.vresolution, env.hresolution))
        next_state = np.append(next_state, initial_state[:, :(FRAME_SKIP-1), :, :], axis=1)
        
        reward = agent.get_rewards()[goal_selected]
        reward = np.clip(reward, -1.0, 1.0)
        sum_reward += reward

        goal_status = agent.get_goal_status()[goal_selected] 

        end_eps = dead or is_done

        if end_eps:
            global_reward += np.clip(info['score'], -1.0, 1.0)
            count_eps += 1

        agent.performer.remember( [initial_state, initial_proprioception], action, reward, [next_state, next_proprioception], end_eps)

        if count_eps >= 10 or goal_status == 1:
            count_eps = 0
            global_reward += goal_status
            global_reward = np.clip(global_reward, -1.0, 1.0)
            print('>>GOAL EPISODE %d. EPS %f. GOAL SELECTED: %d. REWARD %f. LOSS %f'%(agent.gglobal_step, agent.gepsilon, goal_selected, global_reward, LOSS2))
            logg.write('GOAL EPISODE %d. EPS: %f. GOAL SELECTED: %d. REWARD %f. LOSS %f'%(agent.gglobal_step, agent.gepsilon, goal_selected, global_reward, LOSS2))
            if agent.gglobal_step >= N_RANDOM_STEPS2:
                goal_selected = agent.goal_select(initial_state, proprioception=initial_proprioception[0:5])
            else:
                goal_selected = agent.goal_select(initial_state, True, proprioception=initial_proprioception[0:5])

            agent.goal_searcher.remember([initial_state, initial_proprioception[0:5]], goal_selected, global_reward, [next_state, next_proprioception[0:5]], end_eps)
            global_reward = 0.0
            if agent.gglobal_step >=  N_RANDOM_STEPS2:
                LOSS2 = agent.goal_searcher.train(batch_size)
                if agent.gglobal_step % REFRESH_MODEL_NUM2 == 0:
                    agent.goal_searcher.front2back()

        if (agent.global_step >= N_RANDOM_STEPS):
            LOSS += agent.performer.train(batch_size)
            if agent.global_step % REFRESH_MODEL_NUM == 0:
                agent.performer.front2back()

        if not is_done:
            initial_state = next_state
            initial_proprioception = next_proprioception

        if random.random() <= 0.005:
            print("EPISODE: %d. CURRENT GOAL %d. REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f." % (
                i, goal_selected, sum_reward, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step, agent.baredom.min, agent.baredom.max, agent.baredom_freq))
        agent.prev_position = position
        if RENDER:
            env.render()
        
    if (agent.epoch % 1000 == 0):
        agent.performer.save("pmodel%d" % (agent.epoch))
        agent.goal_searcher.save("smodel%d" %(agent.epoch))

    count_loss = agent.step
    if count_loss == 0:
        count_loss = 1
    now = datetime.datetime.now().time()
    print(">>EPISODE: %d. REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f" % (
        i, sum_reward, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step, agent.baredom.min, agent.baredom.max, agent.baredom_freq))
    log.write("EPISODE: %d. REWARDS: %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f" % (
        i, sum_reward, agent.epsilon, agent.step, agent.global_step, LOSS/agent.step, agent.baredom.min, agent.baredom.max, agent.baredom_freq))
    log.flush()
    LOSS = 0.0
