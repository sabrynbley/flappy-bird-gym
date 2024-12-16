import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the NumPy array from the file
# Replace 'data.npy' with the path to your saved NumPy file
data = np.load('reward_testing_turns.npy')

# Step 2: Plot the data
plt.figure(figsize=(8, 6))
plt.plot(data, color='blue')

plt.ylim(bottom=0)

# Step 3: Customize the plot
plt.title('Reward Agent Testing Episode Length')
plt.xlabel('Episode')
plt.ylabel('Number of Turns')
plt.grid(True)

# Step 4: Show the plot
plt.show()