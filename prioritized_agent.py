import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim

# -------------------------------------
# Prioritized Replay Memory Class
# -------------------------------------
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        """
        A simple Prioritized Experience Replay buffer using a list-based approach.
        
        :param capacity: Maximum size of the replay buffer.
        :param alpha: How much prioritization is used (0 = no prioritization, 1 = full prioritization).
        """
        self.capacity = capacity
        self.alpha = alpha
        
        self.memory = []
        self.priorities = []
        self.position = 0  # next insertion index

    def push(self, transition):
        """
        Add a new transition into the buffer, with a default high priority so it gets sampled soon.
        
        :param transition: Dictionary containing {s_t, a_t, s_a, r_t+1, s_t+1, done}
        """
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        # If the memory buffer isn't full yet
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_priority)
        else:
            # Overwrite the oldest transition
            self.memory[self.position] = transition
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions, returning transitions + indices + importance sampling weights.
        
        :param batch_size: Number of experiences to sample.
        :param beta: Importance sampling exponent. 
        :return: (transitions, indices, weights)
        """
        # If the buffer hasn't filled up yet, sample from len(self.memory)
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()  # normalize

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Compute importance sampling (IS) weights
        # w_i = (N * p_i)^(-beta) / max(weights)
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize by the maximum weight in the batch

        transitions = [self.memory[idx] for idx in indices]
        return transitions, indices, weights

    def update_priorities(self, indices, td_errors):
        """
        Update the priorities of sampled transitions based on new TD errors.
        
        :param indices: Indices of the sampled transitions in the replay buffer.
        :param td_errors: The absolute TD error for each transition.
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5  # small constant to avoid zero priority

    def __len__(self):
        return len(self.memory)


# -------------------------------------
# The MLP Model
# -------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x


# -------------------------------------
# The Agent
# -------------------------------------
class Agent:
    def __init__(self, config):
        """
        :param config: A dictionary specifying learning details.
        """
        self.config = config

        # hyperparams for prioritized replay
        self.alpha = config.get('pr_alpha', 0.6)        # how much prioritization is used
        self.beta_start = config.get('pr_beta_start', 0.4) 
        self.beta_frames = config.get('pr_beta_frames', 100000)  # how quickly beta goes to 1
        self.beta = self.beta_start  # this will be updated over training steps
        
        self.Q = MLP(input_size=3, hidden_size=self.config['hidden_size']) 
        self.Q_prime = copy.deepcopy(self.Q)
        self.criterion = torch.nn.MSELoss(reduction='none')  # We'll handle weighting manually
        self.optimizer = optim.Adam(self.Q.parameters(), lr=self.config['alpha'])

        # Create Prioritized Replay
        self.memory = PrioritizedReplayMemory(capacity=self.config['buffer size'], alpha=self.alpha)
        
        self.gamma = self.config['gamma']
        self.train_step_counter = 0  # used to track how many steps so far (for beta annealing)

    def Q_reset(self):
        """
        Reset the MLP to random initial parameters. (Unused in code snippet but included for completeness)
        """
        self.Q = MLP(input_size=3, hidden_size=self.config['hidden_size'])

    def update_Q_prime(self):
        """
        Update the target network by copying the weights from the main network.
        """
        self.Q_prime = copy.deepcopy(self.Q)

    def make_options(self, s_t):
        """
        Create the state-action pairs for both possible actions: 0 or 1.

        :param s_t: a numpy array or list for the state
        :return: Torch tensor of shape [2, 3]
        """
        s_tA = []
        for a in [0, 1]:
            state_action = np.concatenate((s_t, np.array([a])))
            s_tA.append(state_action)
        return torch.tensor(s_tA, dtype=torch.float32)

    def epsilon_t(self, count, n_episodes):
        """
        A dynamic epsilon scheduling approach. You can tweak these rules as you see fit.
        """
        if count <= self.config['epsilon_burnin']:
            return 1.0
        elif self.config['epsilon_burnin'] < count <= self.config['epsilon_burnin2']:
            return 0.5
        else:
            return 1 / ((n_episodes + 1) ** 0.5)

    def pi(self, s_t, epsilon):
        """
        Epsilon-greedy policy based on Q-values. 
        """
        if np.random.rand() <= epsilon:
            return np.random.choice([0, 1])
        else:
            q_values = self.Q(self.make_options(s_t))
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a single transition in memory. 
        Transition is a dict matching the original code format.
        """
        transition = dict()
        transition['s_t'] = state.tolist()
        transition['a_t'] = [action]
        transition['s_a'] = np.array(transition['s_t'] + transition['a_t'])
        transition['r_t+1'] = reward
        transition['s_t+1'] = next_state
        transition['done'] = done
        self.memory.push(transition)

    def sample_batch(self):
        """
        Sample a batch of transitions from prioritized replay memory.
        Returns (batch, indices, weights).
        """
        transitions, indices, weights = self.memory.sample(self.config['B'], beta=self.beta)
        
        # Now parse out (X, y) from these transitions
        X = []
        y = []
        for d in transitions:
            X.append(d['s_a'])
            y_t = d['r_t+1']
            if not d['done']:
                max_a_Q = float(self.Q_prime(self.make_options(d['s_t+1'])).max())
                y_t = y_t + self.gamma * max_a_Q
            y.append(y_t)
        return X, y, indices, weights

    def update_Q(self):
        """
        Fetch a sample from replay memory, then update Q and memory priorities.
        """
        if len(self.memory) < self.config['B']:
            return  # not enough samples to train

        X, y, indices, weights = self.sample_batch()
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        weights_t = torch.tensor(weights, dtype=torch.float32).view(-1, 1)

        # Forward pass
        pred = self.Q(X_t)
        
        # Compute per-sample MSE 
        loss_per_sample = self.criterion(pred, y_t)  # shape = [batch_size, 1]
        
        # Weighted loss: multiply each sample's MSE by its importance-sampling weight
        loss = (loss_per_sample * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in replay buffer
        td_errors = (pred.detach() - y_t).cpu().numpy().reshape(-1)
        self.memory.update_priorities(indices, td_errors)

    def anneal_beta(self):
        """
        Slowly increase beta from beta_start to 1 over beta_frames steps.
        This is typically done every training step or episode.
        """
        self.train_step_counter += 1
        fraction = min(float(self.train_step_counter) / self.beta_frames, 1.0)
        self.beta = self.beta_start + fraction * (1.0 - self.beta_start)