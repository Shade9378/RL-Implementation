import gym
import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
    )
    def forward(self, x):
        return self.net(x)
    
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net,
        preprocessor=ptan.agent.
        float32_preprocessor,
        apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    done_episodes = 0
    batch_episodes = 0
    cur_rewards = []
    
    batch_states, batch_actions, batch_qvals = [], [], []
    ep_states, ep_actions, ep_rewards = [], [], []

    for step_idx, exp in enumerate(exp_source):
        ep_states.append(exp.state)
        ep_actions.append(int(exp.action))
        ep_rewards.append(exp.reward)

        if exp.last_state is None:
            ep_qvals = calc_qvals(ep_rewards)

            batch_states.extend(ep_states)
            batch_actions.extend(ep_actions)
            batch_qvals.extend(ep_qvals)

            ep_states.clear(); ep_actions.clear(); ep_rewards.clear()
            batch_episodes += 1


        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f,"\
                "episodes: %d" % (step_idx, reward,
                mean_rewards,
                done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards,
                step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)

            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" %
                    (step_idx, done_episodes))
                break

            if batch_episodes < EPISODES_TO_TRAIN:
                continue

            optimizer.zero_grad()
            states_v = torch.from_numpy(np.array(batch_states, dtype=np.float32))
            batch_actions_t = torch.tensor(batch_actions, dtype=torch.long)
            batch_qvals_v = torch.tensor(batch_qvals, dtype=torch.float32)

            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            idx = torch.arange(states_v.size(0))
            log_prob_actions_v = batch_qvals_v * log_prob_v[idx, batch_actions_t]
            loss_v = -log_prob_actions_v.mean()
            loss_v.backward()
            optimizer.step()

            batch_episodes = 0
            batch_states.clear()
            batch_actions.clear()
            batch_qvals.clear()

    writer.close()