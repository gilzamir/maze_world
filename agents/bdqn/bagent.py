# -*- coding: utf-8 -*-
import random
import numpy as np
import net
import threading as td
import numpy as np
import actions
from PIL import Image
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.optimizers import Adam
from keras import backend as K
import keras
from keras import layers
from keras import Model
import tensorflow as tf
import io
import os
import sys
import math
from keras.optimizers import RMSprop

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

class RingBuf:
    def __init__(self, size):
        self.data = [None]*(size+1)
        self.start = 0
        self.end = 0
        self.maxlen = size

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def sample(buffer, size):
    indices = random.sample(range(len(buffer)), size)
    result = []
    for i in indices:
        result.append(buffer[i])
    return result

def fn_constantbehavioralcontrol(ctime, owner, min, max, initial_value, current_value, target_value):
    dif = current_value - initial_value
    delta = 0.1 if dif > 0 else -0.1
    delta = delta/dif
    return current_value - delta

def fn_linearbehavioralvalue(ctime, owner, min, max, initial_value, current_value):
    c = min + 0.5 * (max - min)
    d = abs(current_value - c)
    if d > 0:
        return 2.0 * (max-min)/d - 1.0
    else:
        return 2.0

def baredom_control(ctime, owner, min, max, initial_value, current_value, target_value):
    owner.delta_time = ctime - owner.start_time
    if owner.delta_time > 100:
        owner.action_count = [0]*7
        owner.start_time = ctime
        owner.baredom_freq = float(owner.action_count[0])/owner.delta_time
    else:
        owner.action_count[owner.last_action] += 1
    return owner.baredom_freq

class BehavioralRange:
    def __init__(self, name,  owner, initial_value, min=0.0, max=1.0, target_value=2.0, fn_value=fn_linearbehavioralvalue, fn_control=fn_constantbehavioralcontrol):
        self.name = name
        self.owner = owner
        self.min = min
        self.max = max
        self.target_value = target_value
        self.initial_value = initial_value
        self.current_value = initial_value
        self.fn_value = fn_value
        self.fn_control = fn_control

    def update(self, ctime):
        self.current_value = self.fn_control(ctime, self.owner, self.min, self.max, self.initial_value, self.current_value, self.target_value)
        return self.fn_value(ctime, self.owner, self.min, self.max, self.initial_value, self.current_value)


class BehavioralEngine:
    def __init__(self, behaviors =  []):
        self.behaviors = behaviors
        self.sum_of_values = 0.0
    
    def update(self, ctime):
        values = {}
        self.sum_of_values = 0.0
        for b in self.behaviors:
            v = b.update(ctime)
            values[b.name] = v
            self.sum_of_values += v
        return values

    def avg_value(self):
        if len(self.behaviors) > 0:
            return self.sum_of_values/len(self.behaviors)
        else:
            return 0.0  


class DQNAgent:
    def __init__(self, state_size, action_size):

        if type(state_size) == tuple:
            self.state_size = state_size
        else:
            self.state_size = (state_size, state_size)
        self.last_loss = 0.0
        self.action_size = action_size
        self.positive_memory = RingBuf(20000)
        self.negative_memory = RingBuf(20000)
        self.neutral_memory = RingBuf(20000)
        self.gamma = 0.99	 # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025  # 0.00025
        self.psize = 0
        self.nsize = 0
        self.ntsize = 0
        self.skip_frames = 4
        self.step = 0
        self.loss = 0.0
        self.count_loss = 1
        self.global_step = 0
        self.contextual_actions = [0, 1, 2, 3, 4, 5, 6]
        self.epoch = 0
        self.mask_actions = np.ones(self.action_size).reshape(1, self.action_size)
        self.replay_is_running = False

        self.baredom_freq = 0.0

        self.graph = tf.get_default_graph()
        self.session = keras.backend.get_session()
        self.model = self._build_model()
        self.model._make_predict_function()
        self.model._make_test_function()
        self.model._make_train_function()
        self.back_model = self._build_model()
        self.back_model._make_predict_function()
        self.back_model._make_test_function()
        self.back_model._make_train_function()

    def reset(self, is_new_epoch=True):
        if is_new_epoch:
            self.epoch += 1
        self.step = 0
        self.last_action = None
        self.action_count = [0] * 7
        self.delta_time = 0
        self.start_time = 0
        self.baredom = BehavioralRange("baredom", self, 0.0, 0.2, 0.5, fn_control=baredom_control)

    def _build_model(self):
        ATARI_SHAPE = (self.skip_frames, self.state_size[0], self.state_size[1])  # input image size to model
        ACTION_SIZE = self.action_size
        # With the functional API we need to define the inputs.
        frames_input = layers.Input(ATARI_SHAPE, name='frames')
        actions_input = layers.Input((ACTION_SIZE,), name='action_mask')
        size = self.state_size[0] * self.state_size[1] * self.skip_frames
        normalize = layers.Lambda( lambda x : x/6.0) (frames_input)
        flattened = layers.Flatten()(normalize)
        reshape = layers.Reshape( (1, size) )(flattened)
        lstm_layer = layers.recurrent.LSTM(32, return_sequences=True)(reshape) 
        hidden = layers.Dense(32)(lstm_layer)
        output = layers.Dense(ACTION_SIZE)(hidden)

        filtered_output = layers.Multiply(name='QValue')([output, actions_input])

        model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
        model.summary()
        #optimizer = RMSprop(lr=self.learning_rate, rho=0.95, epsilon=0.01)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer, loss=huber_loss)
        return model

    def memory_size(self):
        return len(self.positive_memory)+len(self.negative_memory)+len(self.neutral_memory)

    def positive_msize(self):
        return len(self.positive_memory)

    def negative_msize(self):
        return len(self.negative_memory)

    def neutral_msize(self):
        return len(self.neutral_memory)

    def front2back(self):
        self.back_model.set_weights(self.model.get_weights())

    def back2front(self):
        self.model.set_weights(self.back_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        if reward > 0:
            self.positive_memory.append(
                (state, action, reward, next_state, done))
            self.psize += 1
            if (self.psize > self.positive_memory.maxlen):
                self.psize = self.positive_memory.maxlen
        elif reward < 0:
            self.negative_memory.append(
                (state, action, reward, next_state, done))
            self.nsize += 1
            if (self.nsize > self.negative_memory.maxlen):
                self.nsize = self.negative_memory.maxlen
        else:
            self.neutral_memory.append(
                (state, action, reward, next_state, done))
            self.ntsize += 1
            if (self.ntsize > self.neutral_memory.maxlen):
                self.ntsize = self.neutral_memory.maxlen

    def copy_model(self):
        self.model.save('tmp_model')
        return keras.models.load_model('tmp_model')

    def update_internal(self, is_randomic=False):
        self.step += 1
        self.global_step += 1
        if not is_randomic:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def act(self, state, is_randomic = False):
        p = np.random.rand()
        if is_randomic or p <= self.epsilon:
            self.last_action = np.random.choice(np.arange(0, self.action_size))
        else:
            act_values = self.model.predict([state, self.mask_actions])
            self.last_action = np.argmax(act_values[0])
        self.update_internal(is_randomic)
        #self.baredom_value = self.baredom.update(self.step)
        return self.last_action


    def replay(self, batch_size, postask=None):
        p_size = min(self.psize, batch_size)
        n_size = min(self.nsize, batch_size)
        nt_size = min(self.ntsize, batch_size)
        minibatch = []
        if (p_size > 0):
            minibatch += sample(self.positive_memory, p_size)
        if (n_size > 0):
            minibatch += sample(self.negative_memory, n_size)
        if (nt_size > 0):
            minibatch += sample(self.neutral_memory, nt_size)
        random.shuffle(minibatch)
        batch_size = len(minibatch)

        states = np.zeros(
            (batch_size, self.skip_frames, self.state_size[0], self.state_size[1]))
        next_states = np.zeros(
            (batch_size, self.skip_frames, self.state_size[0], self.state_size[1]))
        actions = []
        rewards = []
        dones = []

        targets = np.zeros((batch_size,))

        idx = 0
        for idx, val in enumerate(minibatch):
            states[idx] = val[0]
            next_states[idx] = val[3]
            actions.append(val[1])
            rewards.append(val[2])
            dones.append(val[4])

        actions_mask = np.ones((batch_size, self.action_size))
        next_Q_values = self.back_model.predict([next_states, actions_mask])

        for i in range(batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.gamma * np.amax(next_Q_values[i])

        action_one_hot = get_one_hot(actions, self.action_size)
        target_one_hot = action_one_hot * targets[:, None]

        h = self.model.fit(
            [states, action_one_hot], target_one_hot, epochs=1, batch_size=batch_size, verbose=0)

        self.last_loss = h.history['loss'][0]
        if postask:
            postask(self, self.last_loss)
        return self.last_loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def load_back(self, name):
        self.back_model.load_weights(name)

    def save_back(self, name):
        self.back_model.save_weights(name)
