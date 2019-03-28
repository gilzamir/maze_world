# -*- coding: utf-8 -*-
import random
import numpy as np
from goal_model import Tree, Node
from mind import BrainFunction, BehavioralEngine, ReplayMemory, BehavioralRange, baredom_control, fn_linearbehavioralvalue, fn_constantbehavioralcontrol
import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Dropout, LSTM
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras import layers
from keras import Model
import tensorflow as tf
from keras.optimizers import RMSprop

class DQNAgent:
    def __init__(self, state_size, action_size, proprioception_size=2):

        if type(state_size) == tuple:
            self.state_size = state_size
        else:
            self.state_size = (state_size, state_size)
        self.proprioception_size = proprioception_size
        self.last_loss = 0.0
        self.action_size = action_size
        self.nb_goals = 6
        
        self.gamma = 0.99	 # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99

        self.learning_rate = 0.0005  # 0.00025

        self.skip_frames = 4
        self.step = 0
        self.loss = 0.0
        self.count_loss = 1
        self.global_step = 0
        self.contextual_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        self.epoch = 0
        self.mask_actions = np.ones(self.action_size).reshape(1, self.action_size)
        self.replay_is_running = False
        self.nb_behavior = 1
        self.KEY_CODE = 5
        self.GATE_CODE = 4

        shared_memory_list = [ReplayMemory(100000)]

        #No visual input information: touch, object value, min_var, max_var, var, x, y, z, angle, goal
        self.performer = BrainFunction(model_builder = self._build_model, input_size = 2,
                                    input_shapes=[(self.skip_frames, self.state_size[0], self.state_size[1]), (self.proprioception_size,)],
                                    output_shape=[self.action_size], shared_memory=shared_memory_list)


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
        self.orientation = 0

    def is_to_learning_follow_gate(self):
        return self.isWithKey

    def is_to_learning_get_key(self):
        return (self.get_dist_to_target() < 10) and (not self.isWithKey)
    
    def is_to_learning_follow_key(self):
        return (self.KEY_CODE in self.last_frame) and (not self.isWithKey)

    def is_to_learning_search_key(self):
        return True

    def is_to_learning_eating(self):
        return self.touching_food

    def get_walk_dist(self):
        return abs(self.prev_position[0] - self.position[0])  + abs(self.prev_position[2] - self.position[2])

    def get_dist_to_target(self):
        return abs(self.position[0] - self.target_position[0]) + abs(self.position[2] - self.target_position[2])
    
    def get_dist_to_gate(self):
        return abs(self.position[0] - self.gate_position[0]) + abs(self.position[2] - self.gate_position[2])

    def r_get_score(self): #0
        return np.clip(self.score, -1.0, 1.0)

    def r_get_foods(self): #3
        return np.clip(self.sum_of_rewards, -1.0, 1.0)

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
        tv = self.target_position - self.position
        av = np.array([0,1])
        av[0] = av[0] * np.cos(agent.orientation)
        av[1] = av[1] * np.sin(agent.orientation)


    def r_gate_follow(self): #1
        if self.get_dist_to_gate() < self.prev_dist_to_gate:
            return 1
        elif self.get_dist_to_gate() > self.prev_dist_to_gate:
            return -1
        else:
            return 0

    def r_homeostatic(self):
        return np.tanh(self.baredom_value)

    def get_rewards(self):
        r = np.zeros(self.nb_goals)
        r[0] = self.r_get_score()
        if self.is_to_learning_follow_gate():
            r[1] = self.r_gate_follow()
        if self.is_to_learning_get_key():
            r[2] = self.r_get_key()
        r[3] = self.r_get_foods()
        if self.is_to_learning_follow_key():
            r[4] = self.r_target_follow()
        if self.is_to_learning_search_key():
            r[5] = self.r_search_key()
        return r

    def get_goal_status(self):
        rewards = self.get_rewards()
        result = np.zeros(len(rewards))
        count = 0
        for i, r in enumerate(rewards):
            if r > 0:
                result[i] = 1.0
                count += 1
        return result, count

    def reset(self, is_new_epoch=True):
        if is_new_epoch:
            self.epoch += 1
        self.baredom_freq = 0.0
        self.action_count = [0]*7
        self.start_time = 0
        self.step = 0
        self.gstep = 0
        self.baredom.current_value = 0
        self.baredom.initial_value = 0
        self.baredom.min = np.random.random() * 0.90
        self.baredom.max = min(1.0, (self.baredom.min +  abs(np.random.normal(0.0, 1.0) ) ) )
        self.sum_of_rewards = 0
        self.prev_dist_to_gate = self.get_dist_to_gate()
        self.prev_dist_to_key = self.get_dist_to_target()

    def _build_model(self):
        ATARI_SHAPE = (self.skip_frames, self.state_size[0], self.state_size[1])  # input image size to model
        ACTION_SIZE = self.action_size
        # With the functional API we need to define the inputs.
        frames_input = layers.Input(ATARI_SHAPE, name='frames')
        propriception_input = layers.Input((self.proprioception_size,), name='proprioception')
        actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
        normalize = layers.BatchNormalization()(frames_input)
        reshape = layers.Flatten()(normalize)
        hidden = layers.Dense(128, activation='tanh')(reshape)
        proprioception_hidden = layers.Concatenate()([hidden, propriception_input])
        hidden2 = layers.Dense(128)(proprioception_hidden)
        output = layers.Dense(ACTION_SIZE, activation='tanh')(hidden2)
        filtered_output = layers.Multiply(name='QValue')([output, actions_input])
        model = Model(inputs=[frames_input,  propriception_input, actions_input], outputs=filtered_output)
        model.summary()
        optimizer = RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01)
        
        model.compile(optimizer, loss='mse')
        return model

    def update_internal(self, is_randomic=False):
        self.step += 1
        self.global_step += 1
        if not is_randomic:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def act(self, state, is_randomic = False, proprioception=[0,0,0,0,0]):
        action = 0
        p = np.random.rand()
        if is_randomic or p <= self.epsilon:
            self.update_internal(is_randomic)
            return np.random.choice(np.arange(0, self.action_size))
        else:
            act_values = self.performer.predict([state, np.expand_dims(proprioception, 0), self.mask_actions])
            action = np.argmax(act_values[0])
            self.update_internal(is_randomic)
            self.last_act_values = act_values[0]
            return action