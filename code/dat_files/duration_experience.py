import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def extract_sessions(name):
    csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
    df = pd.read_csv(csv_file_path)

    filtered_df = df[df.iloc[:,0].str.contains('FR1')]
    last_row_index = filtered_df.index[-1]

    new_list = []
    row_numbers = []

    for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
        new_list.append(elem)
        row_numbers.append(index)

    return row_numbers



result_mat = []

name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']

for name in name_list:

    big_row = []

    neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
    df = pd.read_csv(neurons_path)

    row_numbers = extract_sessions(name)

    for session in row_numbers:

        print(name, session)

        time_path = df.iloc[session,2]
        time_df = pd.read_csv(time_path, header=None)

        duration = time_df.iloc[0,-1]
        big_row.append(duration)

    result_mat.append(big_row)


result_df = pd.DataFrame(result_mat)
print(result_df)

means = result_df.mean()
print(means.tolist())



positions = range(1, len(result_df.columns) * (len(result_df.columns) + 1), len(result_df.columns) + 1)

plt.bar(positions, means*1000, width=3, edgecolor='black', color='royalblue')

for i in range(8):
    print(positions[i]*len(result_df))
    print(result_df.iloc[:,i])
    plt.scatter([positions[i]]*len(result_df), result_df.iloc[:,i]*1000, color='none', edgecolors='black')


plt.ylabel('Duration (ms)')
plt.xticks([])
plt.tight_layout()
plt.show()