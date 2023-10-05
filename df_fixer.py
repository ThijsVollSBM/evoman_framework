import pandas as pd

df = pd.read_csv('pickles\mean_gains_explore.csv')

data_series = df['1']

# Create separate Series for columns 1, 4, and 5
column1 = data_series[:9]
column4 = data_series[9:18]
column5 = data_series[18:]

# Print the separate Series
print("Column 1:")
print(column1)
print("\nColumn 4:")
print(column4)
print("\nColumn 5:")
print(column5)

column1.reset_index(drop=True, inplace=True)
column4.reset_index(drop=True, inplace=True)
column5.reset_index(drop=True, inplace=True)

result_df = pd.concat([column1, column4, column5], axis=1)
result_df.columns = ['Enemy1', 'Enemy4', 'Enemy5']

# Print the resulting DataFrame
print(result_df)

result_df.to_csv('pickles/explore_fixed.csv')