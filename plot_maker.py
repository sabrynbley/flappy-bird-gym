import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the NumPy array from the file
# Replace 'data.npy' with the path to your saved NumPy file
data = np.load('reward_testing.npy')

# Step 2: Plot the data
plt.figure(figsize=(8, 6))
plt.plot(data, label='Line Plot', color='blue')

# Step 3: Customize the plot
plt.title('Line Plot from NumPy Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Step 4: Show the plot
plt.show()