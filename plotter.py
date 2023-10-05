import pandas as pd
import matplotlib.pyplot as plt

mode = 'EXPLORE'
enemy = '5'

# Load your CSV data into a DataFrame
data = pd.read_csv(f'pickles/enemy_{enemy}_results_{mode}.csv')

# Calculate the average of mean and max values over 10 runs for each generation
average_mean = data.groupby('generation')['mean'].mean()
average_max = data.groupby('generation')['best'].mean()
std_mean = data.groupby('generation')['mean'].std()
std_max = data.groupby('generation')['best'].std()

# Create a line plot with error bars
plt.figure(figsize=(10, 6))

# Plot average mean with error bars
plt.errorbar(average_mean.index, average_mean, yerr=std_mean, label='Average Mean', capsize=5)

# Plot average max with error bars
plt.errorbar(average_max.index, average_max, yerr=std_max, label='Average Max', capsize=5)

plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.legend()
#plt.title(f'Average Mean and Max Fitness Values with Standard Deviation Across Generations for enemy {enemy}')
plt.grid(True)
plt.show()






