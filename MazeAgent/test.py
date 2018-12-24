import env as e
from env import Environment
import random
import actions

env = Environment()
is_done = False
frame = env.reset()[0]
for i in range(random.randint(0, 30)):
    frame = env.step(1)[0]
while not is_done:
    #action = random.randint(0, len(e.ACTIONS)-1)
    #action = 1
    print("Input action: ")
    action = int(input())
    if action == -1:
        env.reset()
    else:
        frame, lifes, energy, score, isPickUpNear, nearPickUpValue, is_done = env.step(action)
