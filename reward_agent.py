import time
import numpy as np
# import flappy_bird_gym
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
import agent
import pickle

# Create agent
agent_config = {'gamma': 0.1,              # the discount factor
                'train_epsilon': 0.5,            # the epsilon-greedy parameter for training
                'eval_epsilon': 0.1,     # the epsilon-greedy parameter for evaluating
                'alpha': 0.01,            # the learning rate
                'hidden_size': 5,         # the hidden layer size
                'buffer size': 100,     # set the memory size
                'B': 5,                    # set the batch size
                'C': 20,                  # when to update the target approximator
                'n_steps': 4,              # the number of steps to use to update
                'epsilon_burnin': 750}     # when to start burning epsilon value
agent = agent.Agent(agent_config)

# Train agent
print("Training Agent:")
train_ep_score = []
for episode in range(1000):
    env = FlappyBirdEnvSimple(bird_color="red", seed=False)
    state = env.reset()
    score = 0

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action
        action = agent.pi(state, agent.epsilon_t(turns, episode))  #   agent.config['train_epsilon']

        # Evolve environment
        update = env.step(action)
        next_state, reward, done, __ = update
        score += reward
        time.sleep(0.0001)

        # Store experience
        experience_t = dict()
        experience_t['s_t'] = state.tolist()
        experience_t['a_t'] = [action]
        experience_t['s_a'] = np.array(experience_t['s_t'] + experience_t['a_t'])
        experience_t['r_t+1'] = reward
        experience_t['s_t+1'] = next_state
        experience_t['done'] = done
        agent.memory.append(experience_t)

        # Update state
        state = next_state

        if len(agent.memory) > agent_config['buffer size']:
            agent.memory.pop(0)  # remove oldest experience

        if turns % agent_config['n_steps'] == 0:  # time for update...
            batch = agent.make_batch()
            X = batch[0]
            y = batch[1]

            # initial_weights = {name: param.clone() for name, param in agent.Q.named_parameters()} # todo

            agent.update_Q(X, y)

            # Compare updated weights with initial weights
            # for name, param in agent.Q.named_parameters():
            #     print(f"Layer: {name}")
            #     print(f"Initial weights:\n{initial_weights[name]}")
            #     print(f"Updated weights:\n{param.data}")
            #     print(f"Difference:\n{param.data - initial_weights[name]}") # todo

        if turns % agent_config['C'] == 0:  # time to update target approximator
            agent.update_Q_prime()

        if done:
            env.render()
            time.sleep(0.0001)  # change to 0.5

    train_ep_score.append(env.get_game_score())
    if episode % 100 == 0:
        print(episode, "- Score:", np.sum(train_ep_score))

# Save the scores and trained agent to a file
np.save("reward_training.npy", train_ep_score)
with open('reward_agent.pkl', 'wb') as file:
    pickle.dump(agent, file)

# To load the list from the `.npy` file: loaded_list = np.load("baseline_training.npy").tolist()


# Eval agent
print("Testing Agent:")
test_ep_score = []
for episode in range(10):
    env = FlappyBirdEnvSimple(bird_color="red", seed=False)
    state = env.reset()
    score = 0

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action
        action = agent.pi(state, agent.config['eval_epsilon'])

        # Evolve environment
        update = env.step(action)
        next_state, reward, done, __ = update
        score += reward
        time.sleep(0.01)

        # Store experience
        experience_t = dict()
        experience_t['s_t'] = state.tolist()
        experience_t['a_t'] = [action]
        experience_t['s_a'] = np.array(experience_t['s_t'] + experience_t['a_t'])
        experience_t['r_t+1'] = reward
        experience_t['s_t+1'] = next_state
        experience_t['done'] = done
        agent.memory.append(experience_t)

        # Update state
        state = next_state

        if len(agent.memory) > agent_config['buffer size']:
            agent.memory.pop(0)  # remove oldest experience

        if done:
            env.render()
            time.sleep(0.01)

    test_ep_score.append(env.get_game_score())
    if episode % 2 == 0:
        print(episode, "- Score:", np.sum(test_ep_score))

print("Score:", np.sum(test_ep_score))
# Save the list to a `.npy` file
np.save("reward_testing.npy", test_ep_score)

