import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

enemy = '2'
eval = 'static'

# Load your CSV data into a DataFrame
data = pd.read_csv(f'results/enemy_{enemy}_results_cma_{eval}_eval.csv')

# Calculate the average of mean and max values over 10 runs for each generation
average_mean = data.groupby('generation')['mean'].mean()
average_max = data.groupby('generation')['best'].mean()
std_mean = data.groupby('generation')['mean'].std()
std_max = data.groupby('generation')['best'].std()

# Create a line plot with error bars
plt.figure(figsize=(10, 6))

# Interpolate data for smoother edges
interpolation_points = 200
x_smooth = np.linspace(average_mean.index.min(), average_mean.index.max(), interpolation_points)
y_mean_smooth = np.interp(x_smooth, average_mean.index, average_mean)
y_max_smooth = np.interp(x_smooth, average_max.index, average_max)
std_mean_smooth = np.interp(x_smooth, average_mean.index, std_mean)
std_max_smooth = np.interp(x_smooth, average_max.index, std_max)

# Plot the average mean as a filled area (light blue)
plt.fill_between(x_smooth, y_mean_smooth - std_mean_smooth, y_mean_smooth + std_mean_smooth, alpha=0.3, label='Average Mean', color='green')

# Plot the centerline for the average mean (blue)
plt.plot(x_smooth, y_mean_smooth, color='green', linestyle='-', label='Mean Centerline')

# Plot the average max as a filled area (light purple)
plt.fill_between(x_smooth, y_max_smooth - std_max_smooth, y_max_smooth + std_max_smooth, alpha=0.3, label='Average Max', color='orange')

# Plot the centerline for the average max (purple)
plt.plot(x_smooth, y_max_smooth, color='orange', linestyle='-', label='Max Centerline')

plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.legend()
#plt.title(f'Average Mean and Max Fitness Values with Standard Deviation Across Generations for enemy {enemy}')
plt.grid(True)
plt.show()