# Embedded file name: C:\Users\Gilzamir Gomes\lab\maze_world\agents\agent_dqn_hrl_bm4\goal_model.py
# Compiled at: 2019-03-03 15:37:10
# Size of source mod 2**32: 1612 bytes
import numpy as np
from collections import deque

class Node:

    def __init__(self, ID, parent=None, children=None, desc=None):
        self.ID = ID
        if children:
            self.children = children
        else:
            self.children = []
        self.value = 0.0
        self.checked = False
        self.parent = parent
        self.description = desc

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

    def create_node(self, ID, parent, desc=None):
        node = Node(ID, parent, desc=desc)
        if not parent == None:
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
        if not node.checked:
            has_no_checked_children = False
            if len(node.children) > 0:
                for child in node.children:
                    if not child.checked:
                        has_no_checked_children = True
                        stack.append(child)

            if not has_no_checked_children:
                goals.append(node)

    return goals
