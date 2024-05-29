import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from image_function import *
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


interval_list = []

name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']

for name in name_list:

    neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
    df = pd.read_csv(neurons_path)


    row_numbers = extract_sessions(name)

    #row_numbers = row_numbers[6:]
    row_numbers = row_numbers[2:4]

    for session in row_numbers:

        print(name,session)

        dat_path = df.iloc[session,0]
        all_actions_df = number_cycles(dat_path)


        time_num_list = []

        data = load_data(dat_path)
        start_seq, end_seq = new_sequence(data)

        for (_, row1), (_, row2) in zip(start_seq.iterrows(), end_seq.iterrows()):
            val = row2.iloc[7] - row1.iloc[7] + 1
            

            row = [row1.iloc[0], val]
            time_num_list.append(row)

        time_num_df = pd.DataFrame(time_num_list)





        big_counter = 0
        counter = 0
        sequence = ['sequence', 'zone 2', 'pellet', 'zone 1']

        result_list = []

        for index, row in all_actions_df.iterrows():
            #print(row['type'])
            if row['type'] == sequence[counter]:
                if row['type'] == 'sequence':
                    num = time_num_df[time_num_df[0] == row['time']].iloc[0,1]
                    if num >= 1:
                        counter += 1
                        result_list.append(row.tolist())
                    else: 
                        counter = 0

                else:
                    counter += 1
                    result_list.append(row.tolist())
            else:
                counter = 0
            if counter == len(sequence):
                counter = 0
                big_counter += 1



        result_df = pd.DataFrame(result_list)

        if len(result_df) > 0:
            start_list = result_df[result_df.iloc[:,1] == 'sequence'].iloc[:,0].tolist()

            for start_val in start_list:
                sub = time_num_df[time_num_df.iloc[:,0] > start_val]
                if len(sub) > 0:
                    end_val = sub.iloc[0,0]
                    #print(start_val,end_val)
                    
                    interval = end_val - start_val 
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

plt.xlabel('Duration of complete cycles for last two FR5 sessions (ms)')
plt.ylabel('Frequency')

plt.xlim(5000, 70000)
plt.tight_layout()
plt.show()
