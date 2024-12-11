import time
import numpy as np

# import flappy_bird_gym
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

# env = flappy_bird_gym.make("FlappyBird-v0")
env = FlappyBirdEnvSimple(bird_color="blue")
env.reset()
score = 0

done = False
while not done:
    env.render()

    # Getting random action:
    # action = env.action_space.sample()

    if np.random.rand() < 0.25:
        action = 1
    else:
        action = 0
    print("action:", action)

    # Processing:
    update = env.step(action)
    obs, reward, done, __ = update

    score += reward
    print(f"Obs: {obs}\n"
          f"Action: {action}\n"
          f"Score: {score}\n")

    # time.sleep(1 / 30)
    time.sleep(0.05)

    if done:
        env.render()
        time.sleep(0.5)
