from multiprocessing import Pipe, Process
from os import path as osp

import gym

from debug_wrappers import MonitorEnv, NumberFrames

def thunk(env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir):
    env = gym.make(env_id)
    # We calculate the env seed like this so that changing the global seed completely
    # changes the whole set of env seeds.
    env_seed = seed * n_envs + env_n
    env.seed(env_seed)

    if env_n == 0:
        env_log_dir = osp.join(log_dir, "env_0")
    else:
        env_log_dir = None
    env = MonitorEnv(env, "Env {}".format(env_n), log_dir=env_log_dir)

    if debug:
        env = NumberFrames(env)

    env = preprocess_wrapper(env, max_n_noops)

    return env

def make_make_env_fn(env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir):
        return thunk

def make_envs(env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir):
    make_env_fns = [make_make_env_fn(env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir) for env_n in range(n_envs)]
    envs = [SubProcessEnv(make_env_fns[env_n], env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir) for env_n in range(n_envs)]

    return envs


class SubProcessEnv:
    """
    Run a gym environment in a subprocess so that we can avoid GIL and run multiple environments
    asynchronously from a single thread.
    """

    @staticmethod
    def env_process(pipe, make_env_fn, env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir ):
        env = make_env_fn(env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir)
        pipe.send((env.observation_space, env.action_space))
        while True:
            cmd, data = pipe.recv()
            if cmd == 'step':
                action = data
                obs, reward, done, info = env.step(action)
                pipe.send((obs, reward, done, info))
            elif cmd == 'reset':
                obs = env.reset()
                pipe.send(obs)

    def __init__(self, make_env_fn, env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir):
        p1, p2 = Pipe()
        self.pipe = p1
        self.proc = Process(target=self.env_process, args=[p2, make_env_fn, env_n, env_id, preprocess_wrapper, max_n_noops, n_envs, seed, debug, log_dir])
        self.proc.start()
        self.observation_space, self.action_space = self.pipe.recv()

    def reset(self):
        self.pipe.send(('reset', None))
        return self.pipe.recv()

    def step(self, action):
        self.pipe.send(('step', action))
        return self.pipe.recv()

    def close(self):
        self.proc.terminate()
