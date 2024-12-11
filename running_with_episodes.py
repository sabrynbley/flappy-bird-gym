import time
import numpy as np
import flappy_bird_gym

episode_outcomes = []
for _ in range(10):
    env = flappy_bird_gym.make("FlappyBird-v0")
    env.reset()
    score = 0

    done = False
    turns = 0
    while not done:
        turns += 1  # count turns taken
        env.render()

        # Getting random action:
        # action = env.action_space.sample()
        if np.random.rand() < 0.25:
            action = 1
        else:
            action = 0

        # Processing action:
        obs, reward, done, __ = env.step(action)

        score += reward
        # print(f"Obs: {obs}\n"  # Horizontal distance to next pipe, Difference between player's y & next hole's y position
        #       f"Action: {action}\n"  # 1 - flap, 0 - noop (idle)
        #       f"Score: {score}\n"  # sum of rewards through entire game
        #       f"Reward: {reward}\n")  # reward for given state, action

        time.sleep(0.05)

        if done:
            env.render()
            time.sleep(0.5)
            # print("Turns Taken:", turns)  # display number of turns taken by agent

    episode_outcomes.append({'Turns': turns,
                             'Score': score})

for epi in episode_outcomes:
    print("Turns:", epi['Turns'])
    print("Rewards:", epi['Score'], '\n')



