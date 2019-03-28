# Embedded file name: C:\Users\Gilzamir Gomes\lab\maze_world\agents\agent_dqn_hrl_bm4\goal_model.py
# Compiled at: 2019-03-03 15:37:10
# Size of source mod 2**32: 1612 bytes
import numpy as np
from collections import deque

class Node:
    def __init__(self, ID, parent=None, children=None, desc=None, condiction=lambda: True, replay_memory=None):
        self.ID = ID
        if children:
            self.children = children
        else:
            self.children = []
        self.value = 0.0
        self.checked = False
        self.parent = parent
        self.description = desc
        self.condiction = condiction
        self.counter = 0
        self.replay_memory = replay_memory

    def __str__(self):
        return '%d' % self.ID

    def __repr__(self):
        return self.__str__()


class Tree:

    def __init__(self, root):
        self.map = {}
        self.root = root
        self.map[root.ID] = root

    def add_child(self, node, child):
        child.parent = node
        node.children.append(child)
        self.map[child.ID] = child

    def create_node(self, ID, parent, desc=None, condiction=lambda: True):
        node = Node(ID, parent, desc=desc, condiction=condiction)
        if parent:
            parent.children.append(node)
        self.map[ID] = node
        return node

    def get_node(self, ID):
        return self.map[ID]


def get_next_goals(root):
    goals = []
    stack = deque()
    stack.append(root)
    while len(stack) > 0:
        node = stack.pop()
        borda = deque()
        for child in node.children:
            if not child in borda:
                borda.append(child)
        cand = []
        while len(borda) > 0:
            cur_child = borda.pop()
            if cur_child.condiction():
                cand.append(cur_child)
            for child2 in cur_child.children:
                borda.append(child2)
        if len(cand) == 0 and node.condiction():
            goals.append(node)
        else:
            for child in cand:
                stack.append(child)
    return goals

def get_current_goal(goals):
    #print(goals)
    for goal in goals:
        if goal.condiction():
            return goal.ID
    return 0