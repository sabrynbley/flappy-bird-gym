import time
import numpy as np
# import flappy_bird_gym
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
import agent

# Create agent
agent_config = {'gamma': 0.9,              # the discount factor
                'epsilon': 0.5,            # the epsilon-greedy parameter
                'alpha': 0.1,            # the learning rate
                'hidden_size': 64,         # the hidden layer size
                'buffer size': 100,     # set the memory size
                'B': 5,                    # set the batch size
                'C': 20,                  # when to update the target approximator
                'n_steps': 10,              # the number of steps to use to update
                'epsilon_burnin': 10000}       # set up the agent config
agent = agent.Agent(agent_config)

# Train agent
train_ep_score = []
for episode in range(20000):
    env = FlappyBirdEnvSimple(bird_color="blue")
    state = env.reset()
    score = 0

    done = False
    turns = 0
    while not done:
        turns += 1
        env.render()

        # Generate action
        action = agent.pi(state, agent.epsilon_t(turns, episode))

        # Evolve environment
        update = env.step(action)
        next_state, reward, done, __ = update
        reward = 1  # OG reward always = 1
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

        if turns % agent_config['n_steps'] == 0:  # time for update...
            batch = agent.make_batch()
            X = batch[0]
            y = batch[1]
            agent.update_Q(X, y)

        if turns % agent_config['C'] == 0:  # time to update target approximator
            agent.update_Q_prime()

        if done:
            env.render()
            time.sleep(0.1)  # To best watch, set to 0.5

    train_ep_score.append(env.get_game_score())  
    if episode % 100 == 0:
        print(episode, "- Average Score:", np.mean(train_ep_score))



# Eval agent

