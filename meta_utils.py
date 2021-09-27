import gym
import random
import re
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class ActionRepeatWrapper(gym.Wrapper):
    """Gym wrapper for repeating actions."""
    def __init__(self, env, action_repeat, discount):
        gym.Wrapper.__init__(self, env)
        self._env = env
        self._action_repeat = action_repeat
        self._discount = discount

    def reset(self, seed=None):
        return self._env.reset(seed=seed)

    def step(self, action):
        total_reward = 0.0
        discount = 1.0
        for _ in range(self._action_repeat):
            obs, reward, done, info = self._env.step(action)
            total_reward += reward * discount
            discount *= self._discount
            if done:
                break
        return obs, total_reward, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

class FrameStackWrapper(gym.Wrapper):
    """Gym wrapper for stacking image observations."""
    def __init__(self, view, env, k):
        self.view = view
        self._env = env
        gym.Wrapper.__init__(self, env)
        self._k = k
        if str(self.view) == 'both' or self.view == 'double_view_3':
            self._frames1 = deque([], maxlen=k)
            self._frames3 = deque([], maxlen=k)
        else:
            self._frames = deque([], maxlen=k)
        self._proprio_obs_stack = deque([], maxlen=k)
        shp = env.observation_space['im_rgb'].shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space['im_rgb'].dtype)
        self._max_episode_steps = env.max_path_length

    def reset(self, seed=None):
        obs = self._env.reset(seed=seed)
        for _ in range(self._k):
            if str(self.view) == 'both' or self.view == 'double_view_3':
                self._frames1.append(obs['im_rgb1'])
                self._frames3.append(obs['im_rgb3'])
            else:
                self._frames.append(obs['im_rgb'])
            self._proprio_obs_stack.append(obs['proprio'])
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        if str(self.view) == 'both' or self.view == 'double_view_3':
            self._frames1.append(obs['im_rgb1'])
            self._frames3.append(obs['im_rgb3'])
        else:
            self._frames.append(obs['im_rgb'])
        self._proprio_obs_stack.append(obs['proprio'])
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._proprio_obs_stack) == self._k
        proprio_obs = np.concatenate(list(self._proprio_obs_stack), axis=0)
        if str(self.view) == 'both' or self.view == 'double_view_3':
            assert len(self._frames1) == self._k
            assert len(self._frames3) == self._k
            img_obs1 = np.concatenate(list(self._frames1), axis=0)
            img_obs3 = np.concatenate(list(self._frames3), axis=0)
            return img_obs1, img_obs3, proprio_obs
        else:
            assert len(self._frames) == self._k
            img_obs = np.concatenate(list(self._frames), axis=0)
            return img_obs, proprio_obs

    def render(self, **kwargs):
        return self._env.render(**kwargs)