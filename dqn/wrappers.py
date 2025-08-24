import collections
import numpy as np
import cv2
import gymnasium as gym
import gymnasium.spaces as spaces
import ale_py

gym.register_envs(ale_py)

class FireResetEnv(gym.Wrapper):
    """
    For ALE games that need a FIRE action to start. Works with Gymnasium API.
    """
    def __init__(self, env=None):
        super().__init__(env)
        am = env.unwrapped.get_action_meanings()
        assert len(am) >= 3 and am[1] == "FIRE", "Env does not have FIRE action"

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max over last two frames.
    """
    def __init__(self, env=None, skip=4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, t, tr, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            if t or tr:
                terminated, truncated = t, tr
                break
        max_frame = np.maximum.reduce(self._obs_buffer)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info

class ProcessFrame84(gym.ObservationWrapper):
    """
    Convert to 84x84 grayscale (HWC).
    """
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    @staticmethod
    def process(frame):
        assert frame.ndim == 3 and frame.shape[-1] == 3, "Expected RGB frame"
        img = frame.astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized[18:102, :]
        x_t = np.reshape(x_t, (84, 84, 1)).astype(np.uint8)
        return x_t

    def observation(self, obs):
        return ProcessFrame84.process(obs)

class ImageToPyTorch(gym.ObservationWrapper):
    """
    Move channel to first: HWC(84,84,1) -> CHW(1,84,84).
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape  # (84,84,1)
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])  # (1,84,84)
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    """
    Stack last n frames along the channel dimension: (1,84,84) -> (n,84,84)
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = self.observation_space  # (1,84,84)
        low = np.repeat(old_space.low, repeats=n_steps, axis=0)
        high = np.repeat(old_space.high, repeats=n_steps, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=dtype)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Scale to [0,1] float32.
    """
    def __init__(self, env):
        super().__init__(env)
        low = self.observation_space.low.astype(np.float32) / 255.0
        high = self.observation_space.high.astype(np.float32) / 255.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32) / 255.0

def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipEnv(env)
    # Some games (including Pong) start without FIRE; FireReset is safe if FIRE exists.
    meanings = env.unwrapped.get_action_meanings()
    if len(meanings) >= 3 and meanings[1] == "FIRE":
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env