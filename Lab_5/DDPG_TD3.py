'''DLP DDPG Lab'''
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
from cfg import send_message

class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))



class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=[256, 512, 256]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1],hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2],action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


class CriticNet(nn.Module):
    def __init__(self, state_dim = 8, action_dim = 2, hidden_dim = (512, 256)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1),
        )

    def forward(self, x, action):
        x = self.network(torch.cat([x, action], dim=1))
        return x

class DDPG:
    def __init__(self, args):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net_1 = CriticNet().to(args.device)
        self._critic_net_2 = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net_1 = CriticNet().to(args.device)
        self._target_critic_net_2 = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net_1.load_state_dict(self._critic_net_1.state_dict())
        self._target_critic_net_2.load_state_dict(self._critic_net_2.state_dict())
        
        self._actor_opt = torch.optim.AdamW(self._actor_net.parameters(),lr=args.lra)
        self._critic_opt_1 = torch.optim.AdamW(self._critic_net_1.parameters(),lr=args.lrc)
        self._critic_opt_2 = torch.optim.AdamW(self._critic_net_2.parameters(),lr=args.lrc)
        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self._criterion = nn.MSELoss()
        self.args = args

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        with torch.no_grad():
            if noise:
                action = self._actor_net(torch.from_numpy(state).view(1,-1).to(self.device))+\
                       torch.from_numpy(self._action_noise.sample()).view(1,-1).to(self.device)
            else:
                action = self._actor_net(torch.from_numpy(state).view(1,-1).to(self.device))
        return action.cpu().numpy().squeeze()

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state, [int(done)])

    def update(self, total_step):
        self._update_behavior_network(self.gamma, total_step)
        
        self._update_target_network(self._target_actor_net, self._actor_net, self.tau)
        self._update_target_network(self._target_critic_net_1, self._critic_net_2, self.tau)
        self._update_target_network(self._target_critic_net_1, self._critic_net_2, self.tau)

    def _update_behavior_network(self, gamma, total_step):
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        q_value_1 = self._critic_net_1(state, action)
        q_value_2 = self._critic_net_2(state, action)
        with torch.no_grad():
           a_next = self._target_actor_net(next_state)
           noise = torch.randn_like(a_next)
           a_next = a_next + 0.1 * noise
           q_next_1 = self._target_critic_net_1(next_state, a_next)
           q_target_1 = reward + gamma * q_next_1 * (1 - done)
           q_next_2 = self._target_critic_net_2(next_state, a_next)
           q_target_2 = reward + gamma * q_next_2 * (1 - done)
           q_target = torch.min(q_target_1, q_target_2)

        critic_loss_1 = self._criterion(q_value_1, q_target) 
        critic_loss_2 = self._criterion(q_value_2, q_target)
        # raise NotImplementedError
        # optimize critic
        self._actor_net.zero_grad()
        self._critic_net_1.zero_grad()
        self._critic_net_2.zero_grad()
        if total_step % self.args.critic_freq:
            critic_loss_1.backward()
            critic_loss_2.backward()
            self._critic_opt_1.step()
            self._critic_opt_2.step()

        if total_step % self.args.actor_freq:
            action = self._actor_net(state)
            actor_loss = -0.5 * self._critic_net_1(state, action).mean() + -0.5 * self._critic_net_2(state, action).mean()
            self._actor_net.zero_grad()
            actor_loss.backward()
            self._actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_((1-tau)*target.data + tau*behavior.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic_1': self._critic_net_1.state_dict(),
                    'critic_2': self._critic_net_2.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic_1': self._target_critic_net_1.state_dict(),
                    'target_critic_2': self._target_critic_net_2.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt_1': self._critic_opt_1.state_dict(),
                    'critic_opt_2': self._critic_opt_2.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic_1': self._critic_net_1.state_dict(),
                    'critic_2': self._critic_net_2.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net_1.load_state_dict(model['critic_1'])
        self._critic_net_2.load_state_dict(model['critic_2'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net_1.load_state_dict(model['target_critic_1'])
            self._target_critic_net_2.load_state_dict(model['target_critic_2'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt_1.load_state_dict(model['critic_opt_1'])
            self._critic_opt_2.load_state_dict(model['critic_opt_2'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    best_reward = float('-inf')
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                break
        if total_steps >= args.warmup and total_reward > best_reward:
            best_reward = total_reward
            send_message('Best DDPG is updata. Reward = ' + str(int(total_reward)) + '.')
            agent.save(model_path = './ddpg_best.pth')
        if episode % 100 ==0:
            agent.save(model_path = './ddpg_' + str(episode) + '.pth')
    env.close()

def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            action = agent.select_action(state,noise=False)
            # execute action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print(f'total reward: {total_reward:.2f}')
                rewards.append(total_reward)
                break
    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ddpg.pth')
    parser.add_argument('--logdir', default='log/ddpg')
    # train
    parser.add_argument('--warmup', default=20000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    parser.add_argument('--critic_freq', default=4, type=int)
    parser.add_argument('--actor_freq', default=1000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20000, type=int)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    agent = DDPG(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
