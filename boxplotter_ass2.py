import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a DataFrame from your CSV data
data = pd.read_csv('gain_results.csv')

# Define the dictionary to map "enemy_set" to "enemies"
enemies_dict = {'1': '[4, 7, 8]', '2': '[5, 6, 3]'}

# Replace "enemy_set" values with "enemies" values using the dictionary
data['enemy_set'] = data['enemy_set'].astype(str)  # Ensure the column is treated as a string
data['enemy_set'] = data['enemy_set'].map(enemies_dict)

# Set the style for seaborn
sns.set(style="whitegrid")

# Create a combined boxplot using seaborn
plt.figure(figsize=(10, 6))
plot = sns.boxplot(data=data, x="enemy_set", y="mean", hue="eval", palette="Set2")

# Update the x-axis label
plot.set(xlabel='enemies')


plt.show()






