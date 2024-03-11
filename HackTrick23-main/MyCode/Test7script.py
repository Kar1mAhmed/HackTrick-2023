import sys
import numpy as np
import math
import random
import json
import requests

import gym
import gym_maze
from gym_maze.envs.maze_manager import MazeManager
from riddle_solvers import *
import pygame
from collections import deque


import torch
import time
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(torch.cuda.current_device())



sample_maze = np.load("hackathon_sample.npy")
agent_id = "9" # add your agent id here
    
manager = MazeManager()
manager.init_maze(agent_id, maze_cells=sample_maze)
env = manager.maze_map[agent_id]



class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.L1 = nn.Linear(self.input_dim, 32)
        self.L2 = nn.Linear(32, 64)
        self.L3 = nn.Linear(64, 64)
        self.L4 = nn.Linear(64, 32)
        self.L5 = nn.Linear(32, 16)
        self.L6 = nn.Linear(16,self.output_dim)
        
        

    def forward(self, x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))
        x = F.relu(self.L5(x))
        x = F.relu(self.L6(x))
        x = F.softmax(x, dim=1)
        return x

    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # setting the state to tensor with the input shape
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def fix_state(state):
    state0 = state[0]
    state1 = state[1]
    state2 = state[2]
    
    ans = []
    for i in state0:
        ans.append(i)
    for i in state1:
        ans.append(i)
    for i in state2:
        for j in i:
            ans.append(j)
    return np.array(ans)





def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    scores_deque = deque(maxlen=100)
    scores = []
    agent_id = '9'
    Actions = ['N', 'S', 'E', 'W']
    for episode in range(1, n_training_episodes + 1):
        state = manager.reset(agent_id)
        saved_log_probs = []
        rewards = []
        state = fix_state(state)
        solved_riddles = []
        for t in range(max_t):
            
            action, log_prob = policy.act(state)
            
                
            saved_log_probs.append(log_prob)
            
            
            action = Actions[action]
            last_position = state[:2]
            state, reward, terminated,truncated, info = manager.step(agent_id, action)
            state = fix_state(state)
            
            # reward computing
            current_position = state[:2]
            
            if last_position == current_position:
                reward = -0.4
            if info['riddle_type'] != None and info['riddle_type'] not in solved_riddles:
                reward = 1
            else:
                reward = -0.1

            rewards.append(reward)
            if terminated:
                break 
            
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        returns = deque(maxlen=max_t) 
        n_steps = len(rewards) 
        
        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns)>0 else 0)
            returns.appendleft( gamma*disc_return_t + rewards[t]) 
            
        eps = np.finfo(np.float32).eps.item()

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()
        
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        
        if episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, sum(rewards)))
        
    return scores



s_size = 14
a_size = 4



maze_hyperparameter = {
    "n_training_episodes": 100,
    "n_evaluation_episodes": 10,
    "max_t": 500,
    "gamma": .99,
    "lr": 1e-2,
    "state_space": s_size,
    "action_space": a_size,
}


maze_policy = ActorCritic(maze_hyperparameter["state_space"], maze_hyperparameter["action_space"]).to(device)
maze_optimizer = optim.Adam(maze_policy.parameters(), lr=maze_hyperparameter["lr"])



scores = reinforce( maze_policy,
                    maze_optimizer,
                    maze_hyperparameter["n_training_episodes"], 
                    maze_hyperparameter["max_t"],
                    maze_hyperparameter["gamma"], 
                    10)