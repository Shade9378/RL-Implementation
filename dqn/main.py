import argparse
import time
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import wrappers
import model

import os
import shutil

DEFAULT_ENV_NAME = "ALE/Pong-v5"
MEAN_REWARD_BOUND = 19.0

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

Experience = collections.namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[i] for i in idxs])
        return (
            np.array(states, copy=False),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states, copy=False),
        )

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        s, _ = self.env.reset()
        self.state = s
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.expand_dims(self.state, axis=0)  # (B,C,H,W)
            state_v = torch.tensor(state_a, dtype=torch.float32, device=device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        self.total_reward += reward
        self.exp_buffer.append(Experience(self.state, action, reward, done, new_state))
        self.state = new_state

        if done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(states, dtype=torch.float32, device=device)
    actions_v = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)
    done_mask = torch.tensor(dones, dtype=torch.bool, device=device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32, device=device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = rewards_v + GAMMA * next_state_values

    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help=f"Env name, default={DEFAULT_ENV_NAME}")
    parser.add_argument("--render", default=False, action="store_true", help="Render human window")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env, render_mode=("human" if args.render else None))
    net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net.load_state_dict(net.state_dict())

    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts + 1e-8)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = float(np.mean(total_rewards[-100:]))

            print(f"{frame_idx}: done {len(total_rewards)} games, "
                  f"reward {m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s")

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed_fps", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_m_reward is None or m_reward > best_m_reward:
                checkpoint_dir = "checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)

                safe_env_name = args.env.replace("/", "_")
                filename = f"{safe_env_name}-best_{m_reward:.0f}.dat"

                torch.save(net.state_dict(), os.path.join(checkpoint_dir, filename))

                # safe_env_name = args.env.replace("/", "_") 
                # torch.save(net.state_dict(), f"{safe_env_name}-best_{m_reward:.0f}.dat")

                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                best_m_reward = m_reward

            if m_reward > MEAN_REWARD_BOUND:
                print(f"Solved in {frame_idx} frames!")
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()

    writer.close()
