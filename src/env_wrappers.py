import gym
import numpy as np
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

"""
Credits: Environment wrappers and usage taken from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
# Very useful wrapper for Mujoco environments since state values can vary by a few magnitudes
class VecNormalize(VecNormalize_): 
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs
    
    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs[-1], rews[-1], news[-1], infos[-1] # Indexing at -1 since we will only ever have 1 env in wrapper
    
    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)[-1] # Indexing at -1 since we will only ever have 1 env in wrapper

    def train(self):
        self.training = True

    def eval(self):
        self.training = False # Prevents ob_rms from updating during testing

# Checks whether done was caused by timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)