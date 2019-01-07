import gym
import MazeWorld
import random
import actions


env = gym.make('MazeWorld-v0')
env2 = gym.make('MazeWorld-v0')
is_done = False
frame = env.reset()[0]
frame2 = env2.reset()[0]

while not is_done:
    action = random.randint(0, 5)
    if action == -1:
        env.reset()
        env2.reset()
    else:
        frame, reward, is_done, _ = env.step(action)
        frame2, reward, is_done, _ = env2.step(action)
        
    print(frame)
    print('Score %d'%(reward))
