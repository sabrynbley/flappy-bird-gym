import copy
import random
from collections import deque
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from torch import optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU for the first layer
        x = F.relu(self.fc2(x))  # ReLU for the second layer
        x = self.fc3(x)  # Output layer (no activation because this is a regression problem)
        return x


class Agent:

    def __init__(self, config):
        """
        Set up the constructor

        :param config: A dictionary specifying learning details.
        """
        self.config = config
        self.Q = MLP(input_size=3, hidden_size=self.config['hidden_size'])  # input = len(states) + len(action)
        self.Q.train()  # Set the model to training mode
        self.Q_prime = copy.deepcopy(self.Q)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.config['alpha'])
        self.memory = []  # init the replay buffer

    def Q_reset(self):
        """
        A function reset the MLP to random initial parameters.
        """
        self.Q = MLP(input_size=2, hidden_size=self.config['hidden_size'])

    def update_Q_prime(self):
        """
        A function to update the target network.
        """
        self.Q_prime = copy.deepcopy(self.Q)

    def make_options(self, s_t):
        """A function to create the state action pairs
            Takes:
                s_t -- a list of the state information
            Returns:
                a torch tensor with the first six columns the state information and the last two columns the actions
        """
        s_tA = []  # init a list to hold the state action information
        for a in [0, 1]:  # loop over actions
            state_action = np.concatenate((s_t, np.array([a])))
            s_tA.append(state_action)  # add and record
        return torch.tensor(s_tA).to(torch.float32)

    def epsilon_t(self, count, n_episodes):
        """Lets try out a dynamic epsilon
            Takes:
                count -- int, the number of turns so far
            Returns:
                float, a value for epsilon
        """
        return self.config['epsilon']
        # if count <= self.config['epsilon_burnin']:  # if we're still in the initial period...
        #     return 1  # choose random action for sure
        # else:
        #     return 1 / ((n_episodes + 1) ** 0.5)  # otherwise reduce the size of epsilon

    def pi(self, s_t, epsilon):
        """
        A function to choose actions using Q-values.
            Takes:
                s_t -- a torch tensor with the first 2 columns the state information and the last column the action
                epsilon -- the probability of choosing a random action
        """
        if np.random.uniform() < epsilon:  # if a random action is chosen...
            return np.random.choice(a=range(len([0, 1])))  # return the random action
        else:
            return [0, 1][torch.argmax(self.Q(self.make_options(s_t)).flatten()).item()]   # otherwise return the action with the highest Q-value as predicted by the MLP

    def make_batch(self):
        """
        A function to make a batch from the memory buffer and target approximator.
            Returns:
                a list with the state-action pair at index 0 and the target at index 1
        """
        batch = np.random.choice(self.memory, self.config['B'])  # sample uniformly
        X, y = [], []  # init the state-action pairs and target
        for d in batch:  # loop over all the data collected
            X.append(d['s_a'])  # record the state action pair
            y_t = d['r_t+1']  # compute the target
            if not d['done']:  # if this state didn't end the episode...
                max_a_Q = self.Q_prime(self.make_options(d['s_t+1'])).max().item()
                # max_a_Q = float(max(self.Q_prime(self.make_options(d['s_t+1']))))  # compute the future value using the target approximator
                y_t = y_t + self.config['gamma'] * max_a_Q  # update the target with the future value
            y.append(y_t)  # record the target
        return [X, y]

    def update_Q(self, X, y):
        """
        A function to update the MLP.
            Takes:
                X -- the features collected from the replay buffer
                y -- the targets
        """
        # do the forward pass
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32).view(len(y), 1)

        outputs = self.Q(X)  # pass inputs into the model (the forward pass)
        loss = self.criterion(outputs, y)  # compare model outputs to labels to create the loss

        # do the backward pass
        self.optimizer.zero_grad()  # zero out the gradients
        loss.backward()  # compute gradients
        self.optimizer.step()  # perform a single optimization step

