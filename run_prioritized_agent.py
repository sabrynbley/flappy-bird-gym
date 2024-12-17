import time
import numpy as np
import pickle
# from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

# Hypothetical environment import for context:
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

from prioritized_agent import Agent

# Create agent
agent_config = {
    'gamma': 0.7,                       # discount factor
    'train_epsilon': 0.5,               # epsilon for training (unused directly, we use dynamic)
    'eval_epsilon': 0.01,               # epsilon for evaluation
    'alpha': 0.01,                      # learning rate
    'hidden_size': 50,                  # hidden layer size
    'buffer size': 1000,                # replay buffer capacity
    'B': 10,                            # batch size
    'C': 20,                            # when to update the target Q network
    'n_steps': 5,                       # update frequency
    'epsilon_burnin': 5000,
    'epsilon_burnin2': 15000,
    # Prioritized Replay hyperparameters
    'pr_alpha': 0.6,         # alpha for prioritized sampling (0.0 = uniform, 1.0 = full priority)
    'pr_beta_start': 0.3,    # initial beta for importance sampling
    'pr_beta_frames': 100000 # how many steps before beta = 1.0
}
bird_agent = Agent(agent_config)

RENDER_SPEED = 0.00000001
num_train_episodes = 40000

print("Training Agent (with Prioritized Replay):")
train_ep_score = []
train_turns = []

for episode in range(num_train_episodes):
    env = FlappyBirdEnvSimple(bird_color="red", seed=False)
    state = env.reset()

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action (using dynamic epsilon)
        action = bird_agent.pi(state, bird_agent.epsilon_t(turns, episode))

        # Step environment
        next_state, reward, done, __ = env.step(action)
        time.sleep(RENDER_SPEED)

        # Store transition in prioritized replay
        bird_agent.store_transition(state, action, reward, next_state, done)

        # Update state
        state = next_state

        # Perform a learning step every n_steps
        if turns % bird_agent.config['n_steps'] == 0:
            bird_agent.update_Q()
            bird_agent.anneal_beta()  # update beta

        # Update target network every C steps
        if turns % bird_agent.config['C'] == 0:
            bird_agent.update_Q_prime()

    train_ep_score.append(env.get_game_score())
    train_turns.append(turns)

    if episode % 100 == 0:
        print(episode, "- Score:", np.sum(train_ep_score[-100:]))  # sum or average of recent episodes

# Save training scores
np.save("prioritized_training1.npy", train_ep_score)
np.save("prioritized_training_turns1.npy", train_turns)

# Save agent
with open('prioritized_agent1.pkl', 'wb') as f:
    pickle.dump(bird_agent, f)

# --------------------------------------------------------
# Evaluate the trained agent
# --------------------------------------------------------
with open('prioritized_agent1.pkl', 'rb') as f:
    bird_agent = pickle.load(f)

print("Testing Agent:")
test_ep_score = []
test_turns = []
num_test_episodes = 100

for episode in range(num_test_episodes):
    env = FlappyBirdEnvSimple(bird_color="red", seed=False)
    state = env.reset()

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action (evaluation epsilon is lower)
        action = bird_agent.pi(state, bird_agent.config['eval_epsilon'])

        # Step environment
        next_state, reward, done, __ = env.step(action)
        score = env.get_game_score()
        time.sleep(0.01)

        # (Optionally store transitions even during test, though it might degrade policy if you keep training)
        # If you want a purely testing scenario, you might avoid storing transitions or updating.
        bird_agent.store_transition(state, action, reward, next_state, done)
        if turns % bird_agent.config['n_steps'] == 0:
            bird_agent.update_Q()
            bird_agent.anneal_beta()  # optional if you want ongoing training

        state = next_state

        if done:
            env.render()
            time.sleep(0.01)

    test_ep_score.append(env.get_game_score())
    test_turns.append(turns)

    if episode % 100 == 0:
        print(episode, "- Score:", np.sum(test_ep_score))

print("Final Test Score Sum:", np.sum(test_ep_score))

# Save testing stats
np.save("prioritized_testing1.npy", test_ep_score)
np.save("prioritized_testing_turns1.npy", test_turns)