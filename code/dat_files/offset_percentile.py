import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from behaviors_function import *


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


name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']


interval_list = []

num = 0

for name in name_list:

    row_numbers = extract_sessions(name)
    print(row_numbers)

    row_numbers = row_numbers[:4]    
    #row_numbers = row_numbers[4:]
    print(row_numbers)

    csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
    df = pd.read_csv(csv_file_path)


    for session in row_numbers:

        print(name, session)

        file_path = df.iloc[session,0]


        data = load_data(file_path)
        start_seq, end_seq = new_sequence(data)
        start_seq_time, end_seq_time = start_seq.iloc[:,0].tolist(), end_seq.iloc[:,0].tolist()

        num += len(start_seq_time) - 1

        for i in range(len(start_seq_time) -1):
            interval = start_seq_time[i+1] - start_seq_time[i]
            interval_list.append(interval)


print(np.mean(interval_list))


min_val = np.min(interval_list)
max_val = np.max(interval_list)


interval_size = 500
num_intervals = (max_val - min_val) // interval_size + 1
interval_counts = {min_val + i * interval_size: 0 for i in range(num_intervals)}

for value in interval_list:
    interval = min_val + (value - min_val) // interval_size * interval_size
    interval_counts[interval] += 1

intervals = list(interval_counts.keys())
counts = list(interval_counts.values())

counts = counts/np.sum(counts)

plt.bar(intervals, counts, width=interval_size, align='edge', edgecolor='black', color='royalblue')

plt.xlabel('Interval between 2 action sequences for FR5 sessions (ms)')
plt.ylabel('Frequency')

plt.xlim(5000, 50000)
plt.tight_layout()
plt.show()

