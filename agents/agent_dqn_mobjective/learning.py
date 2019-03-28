# -*- coding: utf-8 -*-
#IDEIAS PARA A PROXIMA VERSAO
# 1) COLOCAR RECOMPENSA PARA OBJETIVO ATINGIDO
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
from goal_model import Tree, Node, get_next_goals, get_current_goal
from mind import BrainFunction, BehavioralEngine, ReplayMemory, BehavioralRange, baredom_control, fn_linearbehavioralvalue, fn_constantbehavioralcontrol

agent = Agent( (20, 20), 7, 9)
for performer in agent.performers:
    performer.front2back()

def pre_processing(observe):
    observe = np.array(observe)
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

LOSS = [0.0]*agent.nb_goals

RENDER = False

REFRESH_MODEL_NUM = 10000
N_RANDOM_STEPS = 50000

MAX_EPSODES = 100000000
NO_OP_STEPS = 30
FRAME_SKIP = 4
NOOP_ACTION = [0,1,2,3,6]
LEVEL_PROBABILITY = [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

env = gym.make('MazeWorld-v0')

log = open('log0.txt', 'w')
logg = open('logg.txt', 'w')

for i in range(MAX_EPSODES):
    frame = env.reset()

    if i > 0 and i % 50 == 0:
        log.close()
        log = open('log%d.txt'%(i), 'w')
    
    if RENDER:
        env.render()

    env.set_level(np.random.choice(LEVELS, p=LEVEL_PROBABILITY))

    is_done = False

    batch_size = 16

    action = 0
    next_state = None
    initial_proprioception = np.array([0, 0, agent.baredom.min, agent.baredom.max, agent.baredom.current_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = 0
    prev_goal = 0
    for _ in range(random.randint(1, NO_OP_STEPS)):
        action = random.sample(NOOP_ACTION,1)[0]
        frame, reward, _, info = env.step(action, FRAME_SKIP)
        agent.last_frame = frame
        position = info['position']
        if not type(agent.prev_position) == list:
            agent.prev_position = position
        agent.position = position
        agent.target_position = info['target_pos']
        agent.last_reward = reward
        agent.isWithKey = info['isWithKey']
        agent.energy = info['energy']
        orientation = info['orientation']
        agent.orientation = orientation
        agent.touching_food = True if info['isPickUpNear'] else False
        initial_proprioception = [ 1 if info['isPickUpNear'] else 0, np.clip(info['nearPickUpValue'], -1, 1), agent.baredom.min,
                                     agent.baredom.max, agent.baredom.current_value,  position[0], position[1],
                                        position[2], orientation]
    
        goal = agent.get_current_goal()

    agent.reset()

    initial_proprioception = np.array(initial_proprioception)
    
    if not env.useRaycasting:
        frame = pre_processing(frame)

    stack_frame = tuple([frame]*agent.skip_frames)
    initial_state = np.stack(stack_frame, axis=2)
    initial_state = np.reshape([initial_state], (1, agent.skip_frames, env.vresolution, env.hresolution))
    dead = False
    sum_reward = 0
    sum_greward = 0
    greward = 0
    
    while not is_done:
        dead = False
        
        performer = agent.performers[goal]

        action = performer.take_action(initial_state, proprioception=initial_proprioception)
        
        agent.last_action = action

        agent.baredom_value = agent.baredom.update(performer.step)

        frame, last_reward, is_done, info = env.step(agent.contextual_actions[action], 8)

        orientation = info['orientation']
        position = info['position']
        score = info['score']
        agent.position = position
        agent.target_position = info['target_pos']
        agent.isWithKey = info['isWithKey']
        agent.energy = info['energy']
        agent.last_reward = last_reward
        agent.last_frame = frame
        agent.touching_food = True if info['isPickUpNear'] else False
        agent.orientation = orientation
        to_target_dir = agent.calc_to_target_dir()

        next_proprioception = [ 1 if info['isPickUpNear'] else 0, np.clip(info['nearPickUpValue'], -1, 1), 
                                agent.baredom.min, agent.baredom.max, agent.baredom_value] + list(position) + [orientation]
        
        next_proprioception = np.array(next_proprioception)

        next_frame = frame
        if not env.useRaycasting:
            next_frame = pre_processing(next_frame)
        
        next_state = np.reshape([next_frame], (1, 1, env.vresolution, env.hresolution))
        next_state = np.append(next_state, initial_state[:, :(FRAME_SKIP-1), :, :], axis=1)

        greward = agent.get_rewards()[goal]

        reward = 0.95 * greward  + 0.05 *  agent.baredom_value
        reward = np.clip(reward, -1.0, 1.0)

        sum_reward += reward
        sum_greward += greward

        end_eps = dead or is_done

        performer.shared_memory[0].remember([initial_state, initial_proprioception], action, reward, [next_state, next_proprioception], end_eps)
        

        if random.random() <= 0.005:
            d = 1 if performer.step == 0 else performer.step
            print("EPISODE: %d. GAOL %d, REWARDS: %f %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f. WALK DIST %f. TARGET APPROX: %f." % (
                 i, goal, sum_greward, sum_reward, performer.epsilon, performer.step, performer.global_step, LOSS[goal]/d, agent.baredom.min, agent.baredom.max, agent.baredom_freq, agent.get_walk_dist(), agent.get_dist_to_target()))

        if (performer.global_step >= N_RANDOM_STEPS):
            LOSS[goal] += performer.train(batch_size)
            if performer.global_step % REFRESH_MODEL_NUM == 0:
                performer.front2back()

        if not is_done:
            initial_state = next_state
            initial_proprioception = next_proprioception
            prev_goal = goal
            goal = agent.get_current_goal()
        
        agent.prev_dist_to_key = agent.get_dist_to_target()
        agent.prev_dist_to_gate = agent.get_dist_to_gate()
        agent.to_target_dir = to_target_dir
        agent.prev_orientation = agent.orientation
        agent.prev_position = position
        if RENDER:
            env.render()

    for g in range(agent.nb_goals):
        performer = agent.performers[g]
        
        if (performer.epoch % 500 == 0):
            performer.save("pmodel%d_%d" % (g, performer.epoch))
        if performer.step >0:
            now = datetime.datetime.now().time()
            d = 1 if performer.step == 0 else performer.step
            print(">>EPISODE: %d. GOAL %d. REWARDS: %f %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f" % (
                i, g, sum_greward, sum_reward, performer.epsilon, performer.step, performer.global_step, LOSS[g]/d, agent.baredom.min, agent.baredom.max, agent.baredom_freq))
            log.write("EPISODE: %d. GOAL %d. REWARDS: %f %f. EPSILON: %f. STEPS: %d. TOTAL STEPS: %d. AVG LOSS: %f. Baredom %f %f %f" % (
                i, g, sum_greward, sum_reward, performer.epsilon, performer.step, performer.global_step, LOSS[g]/d, agent.baredom.min, agent.baredom.max, agent.baredom_freq))            
            log.flush()
        LOSS[g] = 0.0
log.close()
    