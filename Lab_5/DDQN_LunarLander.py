'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from cfg import send_message

class ReplayMemory:
    __slots__ = ['buffer']
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim = 8, action_dim = 4, hidden_dim = [32, 128, 64]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            # nn.Linear(in_features = state_dim, out_features = hidden_dim, bias = True),
            # nn.ReLU(inplace = True),
            # nn.Linear(in_features = hidden_dim, out_features = hidden_dim, bias = True),
            # nn.ReLU(inplace = True),
            # nn.Linear(in_features = hidden_dim, out_features = action_dim, bias = True)
            nn.Linear(in_features = state_dim, out_features = hidden_dim[0], bias = True),
            nn.LayerNorm(hidden_dim[0]),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = hidden_dim[0], out_features = hidden_dim[1], bias = True),
            nn.LayerNorm(hidden_dim[1]),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = hidden_dim[1], out_features = hidden_dim[2], bias = True),
            nn.LayerNorm(hidden_dim[2]),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = hidden_dim[2], out_features = action_dim, bias = True),
            nn.ReLU(inplace = True),
        )

    def forward(self, input):
        output = self.network(input)
        return output


class DQN:
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq
        self.lr = args.lr

        self._behavior_net = Net().to(self.device)
        self._target_net = Net().to(self.device)
        self._target_net.load_state_dict(self._behavior_net.state_dict())

        self._optimizer = torch.optim.AdamW(self._behavior_net.parameters(), lr = self.lr)
        self._criterion = nn.MSELoss()
        #self._scheduler = CosineAnnealingWarmupRestarts(self._optimizer, \
        #first_cycle_steps = args.episode, cycle_mult = 1.0, max_lr = 0.008, min_lr = 0.0001, warmup_steps = args.episode * 0.3, gamma = 1.0)
        self._memory = ReplayMemory(capacity = args.capacity)

    def select_action(self, state, epsilon, action_space):
        if np.random.rand() <= epsilon:
            action = action_space.sample()
        else:
            with torch.no_grad():
                action = self._behavior_net(torch.tensor(state, dtype = torch.float).to(self.device)).argmax().item()
        return action
    
    def append(self, state, action, reward, next_state, is_done):
        self._memory.append(state, [action], [reward / 10], next_state, [int(is_done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        state, action, reward, next_state, is_done = self._memory.sample(self.batch_size, self.device)
        q_value = self._behavior_net(state).gather(dim = 1, index = action.long())
        
        with torch.no_grad():
            #q_next= self._target_net(next_state).argmax(dim = 1).item()
            action_index = self._behavior_net(next_state).max(dim=1)[1].view(-1,1)
            q_next= self._target_net(next_state).gather(dim=1, index=action_index.long())
            q_target = reward + gamma * q_next * (1 - is_done)

        loss = self._criterion(q_value, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        
    def save(self, model_path, checkpoint = False):
        if checkpoint:
            torch.save({
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                    'scheduler': self._scheduler.state_dict()
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint = False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])
            self._scheduler.load_state_dict(model['scheduler'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    best_reward = float('-inf')

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        
        for t in itertools.count(start=1):
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            
            next_state, reward, is_done, _ = env.step(action)
            agent.append(state, action, reward, next_state, is_done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1

            if is_done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                break

        if total_steps >= args.warmup and total_reward > best_reward:
            best_reward = total_reward
            send_message('Best DQN is updata. Reward = ' + str(int(total_reward)) + '.')
            agent.save(model_path = './dqn_best.pth')

    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    total_steps = 0
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        env.seed(seed)
        state = env.reset()
        total_reward = 0

        for t in itertools.count(start=1):
            action = agent.select_action(state, epsilon, action_space)

            state, reward, is_done, _ = env.step(action)

            total_reward = total_reward + reward
            
            total_steps += 1
            if is_done:
                rewards.append(total_reward)
                writer.add_scalar('Test/Episode Reward', total_reward, total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, n_episode, t, total_reward, epsilon))
                break

    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='dqn.pth')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=300, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=100, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=107324, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
