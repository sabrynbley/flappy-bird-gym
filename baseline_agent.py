import time
import numpy as np
# import flappy_bird_gym
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
import agent
import pickle

RENDER_SPEED = 0.00000001
# Create agent
agent_config = {'gamma': 0.7,              # the discount factor
                'train_epsilon': 0.5,            # the epsilon-greedy parameter for training
                'eval_epsilon': 0.01,     # the epsilon-greedy parameter for evaluating
                'alpha': 0.01,            # the learning rate
                'hidden_size': 50,         # the hidden layer size
                'buffer size': 1000,     # set the memory size
                'B': 10,                    # set the batch size
                'C': 20,                  # when to update the target approximator
                'n_steps': 5,              # the number of steps to use to update
                'epsilon_burnin': 18000}     # when to start burning epsilon value
bird_agent = agent.Agent(agent_config) #TODO
# with open('baseline_agent.pkl', 'rb') as file:
#     bird_agent = pickle.load(file)

# Train agent
print("Training Agent:")
train_ep_score = []
train_turns = []
for episode in range(20000):
    env = FlappyBirdEnvSimple(bird_color="red", seed=False)
    state = env.reset()

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action
        action = bird_agent.pi(state, bird_agent.epsilon_t(turns, episode))  #   bird_agent.config['train_epsilon']

        # Evolve environment
        update = env.step(action)
        next_state, reward, done, __ = update
        if reward > 0:  # if agent didn't crash...
            reward = 1  # it always gets a 1
        time.sleep(RENDER_SPEED)

        # Store experience
        experience_t = dict()
        experience_t['s_t'] = state.tolist()
        experience_t['a_t'] = [action]
        experience_t['s_a'] = np.array(experience_t['s_t'] + experience_t['a_t'])
        experience_t['r_t+1'] = reward
        experience_t['s_t+1'] = next_state
        experience_t['done'] = done
        bird_agent.memory.append(experience_t)

        # Update state
        state = next_state

        if len(bird_agent.memory) > bird_agent.config['buffer size']:
            bird_agent.memory.pop(0)  # remove oldest experience

        if turns % bird_agent.config['n_steps'] == 0:  # time for update...
            batch = bird_agent.make_batch()
            X = batch[0]
            y = batch[1]

            bird_agent.update_Q(X, y)

        if turns % bird_agent.config['C'] == 0:  # time to update target approximator
            bird_agent.update_Q_prime()

        if done:
            env.render()
            time.sleep(RENDER_SPEED)  # change to 0.5

    train_ep_score.append(env.get_game_score())
    train_turns.append(turns)
    if episode % 100 == 0:
        print(episode, "- Score:", np.sum(train_ep_score))

# Save the scores and trained agent to a file
np.save("baseline_training.npy", train_ep_score)
np.save("baseline_training_turns.npy", train_turns)
with open('baseline_agent.pkl', 'wb') as file:
    pickle.dump(bird_agent, file)

# To load the list from the `.npy` file: loaded_list = np.load("baseline_training.npy").tolist()

with open('baseline_agent.pkl', 'rb') as file:
    bird_agent = pickle.load(file)

# Eval agent
print("Testing Agent:")
test_ep_score = []
test_turns = []
for episode in range(100):
    env = FlappyBirdEnvSimple(bird_color="red", seed=False)
    state = env.reset()
    score = 0

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action
        action = bird_agent.pi(state, bird_agent.config['eval_epsilon'])

        # Evolve environment
        update = env.step(action)
        next_state, reward, done, __ = update
        if reward > 0:  # if agent didn't crash...
            reward = 1  # it always gets a 1

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
        bird_agent.memory.append(experience_t)

        # Update state
        state = next_state

        if len(bird_agent.memory) > bird_agent.config['buffer size']:
            bird_agent.memory.pop(0)  # remove oldest experience

        if done:
            env.render()
            time.sleep(0.01)

    test_ep_score.append(env.get_game_score())
    test_turns.append(turns)
    if episode % 100 == 0:
        print(episode, "- Score:", np.sum(test_ep_score))

print("Score:", np.sum(test_ep_score))
# Save the list to a `.npy` file
np.save("baseline_testing.npy", test_ep_score)
np.save("baseline_testing_turns.npy", test_turns)

