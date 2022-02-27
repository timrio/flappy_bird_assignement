import argparse
import os, sys
import gym
import time
import numpy as np
import text_flappy_bird_gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

class SarsaAgent():
    def agent_init(self, agent_init_info):
        self.num_actions = agent_init_info["num_actions"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_info["seed"])
        self.q = {}

        
    def agent_start(self, state):
        current_q = self.q.setdefault(state,[0,0])
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q) # greedy action selection
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, state):
        
        # Choose action using epsilon greedy.
        current_q = self.q.setdefault(state,[0,0])
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        best_q = np.max(current_q)
        number_of_greedy_actions = np.sum(current_q==best_q)
        proba_non_greedy = (self.epsilon / self.num_actions)
        proba_greedy = ((1 - self.epsilon) / number_of_greedy_actions) + (self.epsilon / self.num_actions)
        expected_q = 0 
        for a in range(self.num_actions):
            if current_q[a] != best_q: 
                expected_q += current_q[a] * proba_non_greedy 
            else: 
                expected_q += current_q[a] * proba_greedy
        
        previous_values_list = self.q[self.prev_state]
        previous_values_list[self.prev_action] += self.step_size*(reward + self.discount*expected_q - self.q[self.prev_state][self.prev_action])
        self.q[self.prev_state,self.prev_action] = previous_values_list

        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        previous_values_list = self.q[self.prev_state]
        previous_values_list[self.prev_action] += self.step_size*(reward - self.q[self.prev_state][self.prev_action])
        self.q[self.prev_state,self.prev_action] = previous_values_list
    
        
    def argmax(self, q_values):

        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

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
    sarsa_agent = SarsaAgent.load(args.file)

    # initialise env
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs = env.reset()
    action = sarsa_agent.agent_start(obs)

    while True:
        obs, reward, done, info = env.step(action)
        # Render the game
        os.system("clear")
        sys.stdout.write(env.render())
        time.sleep(0.001) # FPS
    

        sys.stdout.flush()
        if done:
            sarsa_agent.agent_end(reward)
            break
        else:
            action = sarsa_agent.agent_step(reward, obs)   