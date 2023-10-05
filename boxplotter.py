import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

df1 = pd.read_csv('pickles/explore_fixed.csv')
df2 = pd.read_csv('pickles/exploit_fixed.csv')


# Concatenate the two DataFrames vertically
combined_df = pd.concat([df1, df2])

# Create a new column 'Pair' to distinguish the two sets of data
combined_df['Pair'] = ['Explore'] * len(df1) + ['Exploit'] * len(df2)

# Melt the DataFrame to make it suitable for boxplot
melted_df = pd.melt(combined_df, id_vars=['Pair'], value_vars=['Enemy1', 'Enemy4', 'Enemy5'], var_name='Enemy')

# Create boxplots using seaborn
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted_df, x='Enemy', y='value', hue='Pair', palette='Set1')

# Set plot title and labels
plt.title('Individual gain of Explore(red) and Explot(blue) for enemy 1, 4 and 5')
plt.ylabel('')

# Calculate the average for each column in each DataFrame
avg_df1 = df1[['Enemy1', 'Enemy4', 'Enemy5']].mean()
avg_df2 = df2[['Enemy1', 'Enemy4', 'Enemy5']].mean()

# Perform t-tests for each column (pair)
ttest_results = {}
for column in avg_df1.index:
    t_stat, p_value = stats.ttest_ind(df1[column], df2[column],equal_var=False)
    ttest_results[column] = {
        't_statistic': t_stat,
        'p_value': p_value
    }

# Create a DataFrame to display the t-test results
ttest_df = pd.DataFrame(ttest_results).T

# Print the average values and t-test results
print("Averages for CSV File 1:")
print(avg_df1)
print("\nAverages for CSV File 2:")
print(avg_df2)
print("\nT-Test Results:")
print(ttest_df)