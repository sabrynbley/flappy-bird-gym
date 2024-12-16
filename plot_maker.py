import numpy as np
import matplotlib.pyplot as plt

# # Step 1: Load the NumPy array from the file
# data = np.load('baseline_testing_turns.npy')
#
# # Step 2: Plot the data
# plt.figure(figsize=(8, 6))
# # plt.bar(range(len(data)), data, color='blue')
# plt.plot(data, color='blue')
#
# plt.ylim(bottom=0)
#
# # Step 3: Customize the plot
# # plt.title('Baseline Agent Training Scores')
# plt.title('Baseline Agent Testing Episode Length')
# plt.xlabel('Episode')
# # plt.ylabel('Score')
# plt.ylabel('Number of Turns')
# plt.grid(True)
#
# # Step 4: Show the plot
# plt.show()



#  MAKE COMPARISON PLOT
base = np.load('baseline_testing_turns.npy')
reward = np.load('reward_testing_turns.npy')
seed = np.load('seed_testing_turns.npy')
# TODO pr = np.load('pr_testing_turns.npy')

plt.plot(base, label='Baseline', color='blue')
plt.plot(reward, label='Reward', color='orange')
plt.plot(seed, label='Training Seed', color='green')
# TODO plt.plot(pr, label='Priority Replay', color='pink')

plt.ylim(bottom=90)

plt.title('Agent Testing Episode Length Comparison')
plt.xlabel('Episode')
plt.ylabel('Number of Turns')
plt.legend(title='Agents')
plt.grid(True)

plt.show()