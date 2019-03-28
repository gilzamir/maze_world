import random
import numpy as np
from PIL import Image
from collections import deque
from keras.optimizers import Adam
from keras import backend as K
import keras

from goal_model import Tree, Node
from ds import RingBuf, sample

def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def fn_constantbehavioralcontrol(ctime, owner, vmin, vmax, initial_value, current_value, target_value):
    dif = current_value - initial_value
    delta = 0.1 if dif > 0 else -0.1
    delta = delta/dif
    return current_value - delta

def fn_linearbehavioralvalue(ctime, owner, vmin, vmax, initial_value, current_value):
    c = vmin + 0.5 * (vmax - vmin)
    d = abs(current_value - c)

    if current_value < vmin:
        return 2.0 * (1.0 - d/(vmax-current_value)) - 1.0
    elif current_value > vmax:
        return 2.0 * (1.0 - d/(current_value-vmin)) - 1.0
    else:
        return 1.0

def baredom_control(ctime, owner, vmin, vmax, initial_value, current_value, target_value):
    owner.delta_time = ctime - owner.start_time
    if owner.delta_time > 300:
        owner.action_count = [0]*7
        owner.start_time = ctime
    else:
        owner.action_count[owner.last_action] += 1
        if owner.delta_time > 50:
            owner.baredom_freq = float(owner.action_count[0] + owner.action_count[1] + owner.action_count[2])/owner.delta_time
    return owner.baredom_freq

class BehavioralRange:
    def __init__(self, name,  owner, initial_value, vmin=0.0, vmax=1.0, target_value=2.0, fn_value=fn_linearbehavioralvalue, fn_control=fn_constantbehavioralcontrol):
        self.name = name
        self.owner = owner
        self.min = vmin
        self.max = vmax
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

class ReplayMemory():
    def __init__(self, mem_size=20000):
        self.psize = 0
        self.nsize = 0
        self.ntsize = 0
        self.positive_memory = RingBuf(mem_size)
        self.negative_memory = RingBuf(mem_size)
        self.neutral_memory = RingBuf(mem_size)

    def sample(self, batch_size):
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
        return minibatch

    def memory_size(self):
        return len(self.positive_memory)+len(self.negative_memory)+len(self.neutral_memory)

    def positive_msize(self):
        return len(self.positive_memory)

    def negative_msize(self):
        return len(self.negative_memory)

    def neutral_msize(self):
        return len(self.neutral_memory)

    def remember(self, inputs, action, reward, next_inputs, done):
        if reward > 0:
            self.positive_memory.append(inputs +  next_inputs + [action, reward, done])
            self.psize += 1
            if (self.psize > self.positive_memory.maxlen):
                self.psize = self.positive_memory.maxlen
        elif reward < 0:
            self.negative_memory.append(inputs +  next_inputs + [action, reward, done])
            self.nsize += 1
            if (self.nsize > self.negative_memory.maxlen):
                self.nsize = self.negative_memory.maxlen
        else:
            self.neutral_memory.append(inputs +  next_inputs + [action, reward, done])
            self.ntsize += 1
            if (self.ntsize > self.neutral_memory.maxlen):
                self.ntsize = self.neutral_memory.maxlen

class BrainFunction:
    def __init__(self, model_builder, input_size, input_shapes, output_shape, gamma = 0.99, shared_memory=[]):
        self.model = model_builder()
        self.model._make_predict_function()
        self.model._make_test_function()
        self.model._make_train_function()
        self.back_model = model_builder()
        self.back_model._make_predict_function()
        self.back_model._make_test_function()
        self.back_model._make_train_function()
        self.shared_memory = [] + shared_memory
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.input_size = input_size
        self.gamma = gamma

    def train_continue(self, batch_size):
        minibatch = []
        for mem in self.shared_memory:
            minibatch += mem.sample(batch_size)
        random.shuffle(minibatch)
        batch_size = len(minibatch)
        inputs = []
        next_inputs = []
        for shape in self.input_shapes:
            inputs.append(np.zeros((batch_size, ) + shape))
            next_inputs.append(np.zeros((batch_size, ) + shape))
        actions = []
        rewards = []
        dones = []
        targets = np.zeros((batch_size, self.output_shape[0]))

        idx = 0
        for idx, val in enumerate(minibatch):
            for i in range(self.input_size):
                inputs[i][idx] = val[i]
            f = 2 * self.input_size
            k = 0
            for i in range(self.input_size, f):
                next_inputs[k][idx] = val[i]
                k += 1
            actions.append(val[f])
            rewards.append(val[f+1])
            dones.append(val[f+2])

        next_Q_values = self.back_model.predict(next_inputs)

        for i in range(batch_size):
            if dones[i]:
                targets[i] = np.array(rewards[i]) * self.output_shape[0]
            else:
                #print(np.array(rewards[i]) * self.output_shape[0] + self.gamma * np.array(next_Q_values[i]))
                targets[i] = np.array(rewards[i]) * self.output_shape[0] + self.gamma * np.array(next_Q_values[i])
                

        h = self.model.fit(
            inputs, targets, epochs=1, batch_size=batch_size, verbose=0)

        last_loss = h.history['loss'][0]
        return last_loss

    def train(self, batch_size):
        minibatch = []
        for mem in self.shared_memory:
            minibatch += mem.sample(batch_size)
        random.shuffle(minibatch)
        batch_size = len(minibatch)
        
        inputs = []
        next_inputs = []
        for shape in self.input_shapes:
            inputs.append(np.zeros((batch_size, ) + shape))
            next_inputs.append(np.zeros((batch_size, ) + shape))

        actions = []
        rewards = []
        dones = []
        targets = np.zeros((batch_size,))

        idx = 0
        for idx, val in enumerate(minibatch):
            for i in range(self.input_size):
                inputs[i][idx] = val[i]
            f = 2 * self.input_size
            k = 0
            for i in range(self.input_size, f):
                next_inputs[k][idx] = val[i]
                k += 1
            actions.append(val[f])
            rewards.append(val[f+1])
            dones.append(val[f+2])

        actions_mask = np.ones((batch_size, self.output_shape[0]))
        next_Q_values = self.back_model.predict(next_inputs +  [actions_mask])

        for i in range(batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.gamma * np.amax(next_Q_values[i])

        action_one_hot = get_one_hot(actions, self.output_shape[0])
        target_one_hot = action_one_hot * targets[:, None]

        h = self.model.fit(
            inputs + [action_one_hot], target_one_hot, epochs=1, batch_size=batch_size, verbose=0)

        last_loss = h.history['loss'][0]
        return last_loss

    def predict(self, input_array):
        return self.model.predict(input_array)

    def back_predict(self, input_array):
        return self.back_model.predict(input_array)

    def front2back(self):
        self.back_model.set_weights(self.model.get_weights())

    def back2front(self):
        self.model.set_weights(self.back_model.get_weights())

    def copy_model(self):
        self.model.save('tmp_model')
        return keras.models.load_model('tmp_model')
   
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def load_back(self, name):
        self.back_model.load_weights(name)

    def save_back(self, name):
        self.back_model.save_weights(name)