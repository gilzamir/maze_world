# -*- coding: utf-8 -*-
import random
import numpy as np
from goal_model import Tree, Node
from mind import BrainFunction, BehavioralEngine, ReplayMemory, BehavioralRange, baredom_control, fn_linearbehavioralvalue, fn_constantbehavioralcontrol
import math

class DQNAgent:
    def __init__(self, state_size, action_size, proprioception_size=2):
        if type(state_size) == tuple:
            self.state_size = state_size
        else:
            self.state_size = (state_size, state_size)
        self.nb_behavior = 1
        self.nb_goals = 6
        self.KEY_CODE = 5
        self.GATE_CODE = 4
        self.skip_frames = 4

        #No visual input information: touch, object value, min_var, max_var, var, x, y, z, angle, goal
        performerScore = BrainFunction(state_size, proprioception_size, action_size, [ReplayMemory(5000)], self.skip_frames)
        performerScore.epsilon_min = 0.05
        performerScore.epsilon_decay =  ((performerScore.epsilon - performerScore.epsilon_min)/10000)
        performerScore.learning_rate = 0.00025  # 0.00025
        performerScore.MAX_RANDOM_STEPS = 1000
        
        
        #performerFollowKey = BrainFunction(state_size, proprioception_size, action_size, [ReplayMemory(10000)], self.skip_frames)
        
        #performerGetKey = BrainFunction(state_size, proprioception_size, action_size, [ReplayMemory(10000)], self.skip_frames)
        
        performerSearchKey = BrainFunction(state_size, proprioception_size, action_size, [ReplayMemory(20000)], self.skip_frames)
        
        #performerEating = BrainFunction(state_size, proprioception_size, action_size, [ReplayMemory(10000)], self.skip_frames)
        #performerEating.epsilon_min = 0.05
        #performerEating.epsilon_decay =  ((performerEating.epsilon - performerEating.epsilon_min)/10000)
        #performerEating.learning_rate = 0.00025  # 0.00025
        #performerEating.MAX_RANDOM_STEPS = 1000
        
        #performerFollowTel = BrainFunction(state_size, proprioception_size, action_size, [ReplayMemory(10000)], self.skip_frames)
        #performerFollowTel.epsilon_min = 0.05
        #performerFollowTel.epsilon_decay =  ((performerFollowTel.epsilon - performerFollowTel.epsilon_min)/10000)
        #performerFollowTel.learning_rate = 0.00025  # 0.00025
        #performerFollowTel.MAX_RANDOM_STEPS = 1000

        #self.performers = [performerEating, performerSearchKey, performerFollowKey, performerGetKey, performerFollowTel, performerScore]
        self.performers = [None]*self.nb_goals
        self.performers[1] = performerSearchKey
        self.baredom = BehavioralRange("baredom", self, 0.0, 0.1, 1.0, fn_control=baredom_control)
        self.last_action = -1
        self.prev_position = None
        self.position = None
        self.prev_dist_to_key = 0
        self.prev_dist_to_gate = 0
        self.target_position = None
        self.energy = 0
        self.baredom_value = 0
        self.score = 0
        self.isWithKey = 0
        self.last_reward = 0.0
        self.gate_position = np.array([262.65, -143.74, 305.13])
        self.MIN_TESTS = 10000
        self.fitness = np.zeros(5)
        self.last_frame = None
        self.touching_food = False
        self.detected_target_counter = 0
        self.to_target_dir = 0.0
        self.orientation = 0
        self.contextual_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.baredom_freq = 0.0
        self.action_count = [0]*7
        self.start_time = 0
        self.prev_orientation = self.orientation
        self.goal_test_counter = 0

    def get_current_goal(self):
        #if self.touching_food:
        #    return 0
        #if self.is_to_learning_search_key():
        return 1
        #if self.is_to_learning_follow_key():
        #    return 2
        #if self.is_to_learning_get_key():
        #    return 3
        #if self.is_to_learning_follow_gate():
        #    return 4
        #return 5

    def get_rewards(self):
        return [self.r_get_foods(), self.r_search_key(), self.r_target_follow(), self.r_get_key(), self.r_gate_follow(), self.r_get_score()]

    def is_to_learning_follow_gate(self):
        return self.isWithKey

    def is_to_learning_get_key(self):
        return (self.get_dist_to_target() < 20) and (not self.isWithKey)
    
    def is_to_learning_follow_key(self):
        return (not self.isWithKey) and (self.KEY_CODE in self.last_frame)

    def is_to_learning_search_key(self):
        return (not self.isWithKey) and (not self.KEY_CODE in self.last_frame)

    def is_to_learning_eating(self):
        return self.touching_food

    def get_walk_dist(self):
        return abs(self.prev_position[0] - self.position[0])  + abs(self.prev_position[2] - self.position[2])

    def get_dist_to_target(self):
        return abs(self.position[0] - self.target_position[0]) + abs(self.position[2] - self.target_position[2])
    
    def get_dist_to_gate(self):
        return abs(self.position[0] - self.gate_position[0]) + abs(self.position[2] - self.gate_position[2])

    def r_get_score(self): #0
        return self.score

    def r_get_foods(self): #3
        return self.last_reward

    def r_get_key(self): #2
        if self.get_dist_to_target() < 5:
            return 1.0 if self.isWithKey else -1.0
        else:
            return 0.0

    def r_target_follow(self): #4
        if self.get_dist_to_target() < self.prev_dist_to_key:
            return 1
        elif self.get_dist_to_target() > self.prev_dist_to_key:
            return -1
        else:
            return 0

    def r_search_key(self): #5
        if self.KEY_CODE in self.last_frame:
            return 1.0
        else:
            return -1.0

    def calc_to_target_dir(self):
        tv = self.target_position - self.position
        tv = np.array([tv[0], tv[2]])
        av = np.array([0.0, 1.0])
        x = av[0]
        y = av[1]
        av[0] = x * np.cos(self.orientation) - y * np.sin(self.orientation)
        av[1] = x * np.sin(self.orientation) + y * np.cos(self.orientation)
        nav = np.linalg.norm(av)
        ntv = np.linalg.norm(tv)
        if nav != 0 and ntv != 0:
            av = av/nav
            tv = tv/ntv 
            v = np.dot(av, tv)
            #print(v)
            return v
        else:
            return 0


    def r_gate_follow(self): #1
        if self.get_dist_to_gate() < self.prev_dist_to_gate:
            return 1
        elif self.get_dist_to_gate() > self.prev_dist_to_gate:
            return -1
        else:
            return 0

    def r_homeostatic(self):
        return np.tanh(self.baredom_value)

    def reset(self, is_new_epoch=True):
        self.baredom_freq = 0.0
        self.action_count = [0]*7
        self.start_time = 0
        for goal in range(self.nb_goals):
            if self.performers[goal] != None:
                self.performers[goal].reset(is_new_epoch)
        self.baredom.current_value = 0
        self.baredom.initial_value = 0
        self.baredom.min = np.random.random() * 0.90
        self.baredom.max = min(1.0, (self.baredom.min +  abs(np.random.normal(0.0, 1.0) ) ) )
        self.sum_of_rewards = 0
        self.prev_dist_to_gate = 0
        self.prev_dist_to_key = 0
        self.to_target_dir = 0.0
        self.prev_orientation = self.orientation
