import os, sys
import gym
import time
import numpy as np
import text_flappy_bird_gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class NN(nn.Module):
    """
    Convolutional Neural Net with 3 conv layers and two linear layers
    """
    def __init__(self, input_shape = 2, n_actions = 2):
        super(NN, self).__init__()

        self.conv_net  = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),
            nn.Conv2d(16, 16, kernel_size=7, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,4)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1), 
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
            
        self.dense = nn.Sequential(
            nn.Linear(16*28*14, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100,  n_actions)
        )

    def forward(self, x):
        x = self.conv_net(x)
        o = self.dense(x)
        return o

class DQNAgent:

    def agent_init(self, agent_init_info, batch_size = 8, max_memory_size = 3000):

        # Store the parameters provided in agent_init_info.
        self.state_shape = agent_init_info["states_shape"]
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.state_space = 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # DQN network  
        self.dqn = NN(2, self.num_actions).to(self.device)
        self.target_network = NN(2, self.num_actions).to(self.device)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.step_size)

        # Create memory
        self.max_memory_size = max_memory_size

        self.STATE_MEM = torch.zeros(max_memory_size, *self.state_shape)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_shape)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.ending_position = 0
        self.num_in_queue = 0
        
        self.memory_sample_size = batch_size
        
        # Learning parameters
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss


    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) #% self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
    
    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]      
        return STATE, ACTION, REWARD, STATE2, DONE
    
    def step(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.epsilon:  
            return torch.tensor([[random.randrange(self.num_actions)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def update_target_network(self):
        self.target_network.load_state_dict(self.dqn.state_dict())
        self.target_network.eval()

    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return
        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)
        
        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a) 
        with torch.no_grad():
            target = REWARD + torch.mul((self.discount * self.target_network(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
        current = self.dqn(STATE).gather(1, ACTION.long())
        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error
        
    def set_epsilon(self, value):
        self.epsilon = value

    @staticmethod
    def load(path):
        obj = pickle.load(open(path,'rb'))
        return(obj)


if __name__ == "__main__":
    # load agent
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="the pickle file with the agent")
    args = parser.parse_args()
    print(args.file)
    agent = DQNAgent.load(args.file)
    

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)   

    state = env.reset()
    state = torch.Tensor([state])
    action = agent.step(state)
    while True:
        # action
        state_next, reward, done, info = env.step(int(action[0][0]))

        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.2) # FPS
        sys.stdout.flush()

        # process output
        state_next = torch.Tensor([state_next])
        reward = torch.tensor([reward]).unsqueeze(0)
        done = torch.tensor([int(done)]).unsqueeze(0)

        # remember and train
        agent.remember(state, action, reward, state_next, done)
        agent.experience_replay()

        # update next state
        state = state_next
        if done:
            break
        else:
            action = agent.step(state)


