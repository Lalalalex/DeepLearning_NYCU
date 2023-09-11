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
from atari_wrappers import wrap_deepmind, make_atari

class ReplayMemory(object):
    def __init__(self, capacity, device, n_frames = 5, frame_size = 84):
        self.capacity = capacity
        self.device = device
        self.n_frames = n_frames
        self.frame_size = frame_size
        self.position = 0
        self.size = 0

        self.states = torch.zeros((capacity, n_frames, frame_size, frame_size), dtype = torch.uint8)
        self.actions = torch.zeros((capacity, 1), dtype = torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype = torch.long)
        self.dones = torch.zeros((capacity, 1), dtype = torch.long)

    def push(self, state, action, reward, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.position, self.size)

    def sample(self, batch_size):
        indices = torch.randint(0, high = self.size, size = (batch_size,))
        sample_states = self.states[indices, :self.n_frames - 1].to(self.device)
        sample_next_states = self.states[indices, 1:].to(self.device)
        sample_actions = self.actions[indices].to(self.device)
        sample_rewards = self.rewards[indices].to(self.device)
        sample_dones = self.dones[indices].to(self.device)
        return sample_states, sample_next_states, sample_actions, sample_rewards, sample_dones
    def __len__(self):
        return self.size


class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size = 8, stride = 4),
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(64, eps = 1e-5, momentum = 0.1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.BatchNorm2d(64, eps = 1e-5, momentum = 0.1),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)

        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.AdamW(self._behavior_net.parameters(), lr = args.lr, eps = 1.5e-4)

        self._memory = ReplayMemory(capacity = args.capacity, device = args.device)

        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq
        self._criterion = nn.MSELoss()

    def select_action(self, state, epsilon, action_space):
        if np.random.rand() <= epsilon:
            return action_space.sample()
        else:
            with torch.no_grad():
                return self._behavior_net(torch.tensor(state, dtype = torch.float).detach().to(self.device).unsqueeze(0)).argmax().item()
        

    def append(self, state, action, reward, done):
        self._memory.push(state, action, reward, done)

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        state, next_state, action, reward, done = self._memory.sample(self.batch_size)
        q_value = self._behavior_net(state).gather(dim = 1, index = action.long())

        with torch.no_grad():
            q_next= torch.max(self._target_net(next_state).detach(), 1)[0].unsqueeze(1)
            q_target = reward + gamma * q_next * (1 - done)
        
        loss = self._criterion(q_value, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])

def frame_process(frame):
    frame = frame.reshape((1, 84, 84))
    frame = torch.from_numpy(frame)
    return frame

def train(args, agent, writer):
    print('Start Training')
    test_time = 0
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, episode_life = True)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    best_reward = 0
    ewma_reward = 0
    state_queue = deque(maxlen = 5)
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for i in range(5):
            state, _, _, _ = env.step(0)
            state = frame_process(state)
            state_queue.append(state)
        state, reward, done, _ = env.step(1) # fire first !!!

        for t in itertools.count(start=1):
            state = torch.cat(list(state_queue))[1:]
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            
            state, reward, done, info = env.step(action)
            state = frame_process(state)
            state_queue.append(state)
            agent.append(torch.cat(list(state_queue)).unsqueeze(0), action, reward, done)

            if total_steps >= args.warmup:
                agent.update(total_steps)

            total_reward += reward

            if total_steps % args.eval_freq == 0:
                """You can write another evaluate function, or just call the test function."""
                avg_reward = test(args, agent, writer, best_reward)
                test_time = test_time + 1
                agent.save(model_path = args.model + "dqn_" + str(test_time) + ".pth")

            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    agent.save(model_path = args.model + 'break_train_final.pth')
    env.close()


def test(args, agent, writer, best_reward):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw)
    action_space = env.action_space
    e_rewards = []
    state_queue = deque(maxlen = 5)
    
    for i in range(args.test_episode):
        
        state = env.reset()
        e_reward = 0
        done = False
        for j in range(5):
            state, _, _, _ = env.step(0)
            state = frame_process(state)
            state_queue.append(state)
        
        while not done:
            state = torch.cat(list(state_queue))[1:]
            time.sleep(0.01)
            #env.render()
            action = agent.select_action(state, args.test_epsilon, action_space)
            state, reward, done, _ = env.step(action)
            state = frame_process(state)
            state_queue.append(state)
            
            e_reward += reward

        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))
    if float(sum(e_rewards)) / float(args.test_episode) > best_reward:
        best_reward = float(sum(e_rewards)) / float(args.test_episode)
        agent.save(model_path = args.model + 'dqn_break_best.pth')
    return float(sum(e_rewards)) / float(args.test_episode)

def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='./dqn_break_model/')
    parser.add_argument('--logdir', default='log/dqn')
    # train
    parser.add_argument('--warmup', default=20000, type=int)
    parser.add_argument('--episode', default=50000, type=int)
    parser.add_argument('--capacity', default=50000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.00825, type=float)
    parser.add_argument('--eps_decay', default=100000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=100000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='./break_train.pth')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=312551116, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    agent = DQN(args)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer, 10000)
    else:
        #agent.load(args.model)
        train(args, agent, writer)
        


if __name__ == '__main__':
    main()
