import argparse
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
import typing as tt
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

import ptan
from ptan.experience import ExperienceFirstLast, VectorExperienceSourceFirstLast
from ptan.common.utils import TBMeanTracker

GAMMA = 0.99
LEARNING_RATE = 1e-4
ENTROPY_BETA = 0.001
BATCH_SIZE = 128
NUM_ENVS = 50
REWARD_STEPS = 4
CLIP_GRAD = 0.5

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False
    
class A2C(nn.Module):
    def __init__(self, input_shape: tt.Tuple[int, ...], n_actions: int):
        super(A2C, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.policy = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.ByteTensor) -> tt.Tuple[torch.Tensor, torch.Tensor]:
        xx = x / 255
        conv_out = self.conv(xx)
        return self.policy(conv_out), self.value(conv_out)
    
def unpack_batch(batch: tt.List[ExperienceFirstLast], net: A2C, device: torch.device, gamma: float, reward_steps: int):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.asarray(exp.state))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.asarray(exp.last_state))

    states_t = torch.FloatTensor(np.asarray(states)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_t = torch.FloatTensor(np.asarray(last_states)).to(device)
        last_vals_t = net(last_states_t)[1]
        last_vals_np = last_vals_t.data.cpu().numpy()[:, 0]
        last_vals_np *= gamma ** reward_steps
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_t = torch.FloatTensor(rewards_np).to(device)
    return states_t, actions_t, ref_vals_t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("--use-async", default=False, action="store_true", help="Use async vector env (A3C mode)")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device(args.dev)
    env_factories = [
        lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
        for _ in range(NUM_ENVS)
    ]
    if args.use_async:
        env = gym.vector.AsyncVectorEnv(env_factories)
    else:
        env = gym.vector.SyncVectorEnv(env_factories)
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    net = A2C(env.single_observation_space.shape,
    env.single_action_space.n).to(device)
    print(net)
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = VectorExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with RewardTracker(writer, stop_reward=18) as tracker:
        with TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                new_rewards = exp_source.pop_total_rewards()

                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                states_t, actions_t, vals_ref_t = unpack_batch(batch, net, device=device, gamma=GAMMA, reward_steps=REWARD_STEPS)
                batch.clear()

                optimizer.zero_grad()
                logits_t, value_t = net(states_t)

                loss_value_t = F.mse_loss(value_t.squeeze(-1), vals_ref_t)

                log_prob_t = F.log_softmax(logits_t, dim=1)
                adv_t = vals_ref_t - value_t.detach()
                log_act_t = log_prob_t[range(BATCH_SIZE), actions_t]
                log_prob_actions_t = adv_t * log_act_t
                loss_policy_t = -log_prob_actions_t.mean()

                prob_t = F.softmax(logits_t, dim=1)
                entropy_loss_t = ENTROPY_BETA * (prob_t * log_prob_t).sum(dim=1).mean()

                loss_policy_t.backward(retain_graph=True)
                grads = np.concatenate([
                    p.grad.data.cpu().numpy().flatten()
                    for p in net.parameters() if p.grad is not None
                ])

                loss_v = entropy_loss_t + loss_value_t
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                loss_v += loss_policy_t

                tb_tracker.track("advantage", adv_t, step_idx)
                tb_tracker.track("values", value_t, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_t, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_t, step_idx)
                tb_tracker.track("loss_policy", loss_policy_t, step_idx)
                tb_tracker.track("loss_value", loss_value_t, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)

