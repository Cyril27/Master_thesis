import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from image_function import *


big_name = 'all'


if big_name == 'I236':
    name_list = ['I2', 'I3', 'I6']
if big_name == 'L023':
    name_list = ['L0', 'L2', 'L3']
if big_name == 'all':
    name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']


shades_blue = ['#1f77b4', '#4c94ff', '#99b3ff']
shades_red = ['#ff4c4c', '#ff8080', '#ff9999', '#ffb3b3', '#ffcccc', '#ffe6e6']

colors = shades_blue + shades_red


def cycles_counter(all_actions_df):

    big_counter = 0
    counter = 0
    sequence = ['sequence', 'zone 2', 'pellet', 'zone 1']

    for index, row in all_actions_df.iterrows():
        #print(row['type'])
        if row['type'] == sequence[counter]:
            counter += 1
        else:
            counter = 0
        if counter == len(sequence):
            counter = 0
            big_counter += 1

    return big_counter

def plot_num_cycles():

    mat = np.zeros([8,len(name_list)])

    for j,name in enumerate(name_list):

        neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
        df = pd.read_csv(neurons_path)

        filtered_df = df[df.iloc[:,0].str.contains('FR1')]
        last_row_index = filtered_df.index[-1]

        new_list = []
        row_numbers = []

        for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
            new_list.append(elem)
            row_numbers.append(index)


        for i,session in enumerate(row_numbers):

            dat_path = df.iloc[session,0]
            all_actions_df = number_cycles(dat_path)

            val = cycles_counter(all_actions_df)

            mat[i,j] = val
            print(name, session)

    means = np.mean(mat, axis=1)
    x_values = np.arange(len(mat))

   
    fig, ax = plt.subplots()
    for i, row in enumerate(mat):
        for j, value in enumerate(row):
            ax.scatter(i+1, row[j], color='white', edgecolors='black', label = [name_list[j]] if i == 0 else "", alpha = 0.8)

    ax.plot(x_values +1, means, color='black', marker='o', linestyle='-')


    ax.set_xlabel('Session no.')
    ax.set_ylabel('No. of complete cycles')

    ax.set_xticks([1,2,3,4,5,6,7,8])
    ax.set_xticklabels(['FR1 Sess.1', 'FR1 Sess.2', 'FR1 Sess.3', 'FR1 Sess.4', 'FR5 Sess.1', 'FR5 Sess.2', 'FR5 Sess.3', 'FR5 Sess.4'], rotation=45)
    plt.tight_layout()

    #plt.legend()
    plt.show()


#plot_num_cycles()

def cycles_counter_conditional(all_actions_df, time_num_df, cond):

    big_counter = 0
    counter = 0
    sequence = ['sequence', 'zone 2', 'pellet', 'zone 1']

    for index, row in all_actions_df.iterrows():
        #print(row['type'])
        if row['type'] == sequence[counter]:
            if row['type'] == 'sequence':
                num = time_num_df[time_num_df[0] == row['time']].iloc[0,1]
                if num >= cond:
                    counter += 1
                else: 
                    counter = 0

            else:
                counter += 1
        else:
            counter = 0
        if counter == len(sequence):
            counter = 0
            big_counter += 1

    return big_counter

def plot_num_cycles_conditional():
    mat = np.zeros([8,len(name_list)])

    for j,name in enumerate(name_list):

        neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
        df = pd.read_csv(neurons_path)

        filtered_df = df[df.iloc[:,0].str.contains('FR1')]
        last_row_index = filtered_df.index[-1]

        new_list = []
        row_numbers = []

        for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
            new_list.append(elem)
            row_numbers.append(index)


        for i,session in enumerate(row_numbers):

            dat_path = df.iloc[session,0]
            all_actions_df = number_cycles(dat_path)

            ######
            time_num_list = []

            data = load_data(dat_path)
            start_seq, end_seq = new_sequence(data)

            for (_, row1), (_, row2) in zip(start_seq.iterrows(), end_seq.iterrows()):
                val = row2.iloc[7] - row1.iloc[7] + 1
                

                row = [row1.iloc[0], val]
                time_num_list.append(row)

            time_num_df = pd.DataFrame(time_num_list)
            ######

            if i <= 3:
                val = cycles_counter_conditional(all_actions_df, time_num_df, 1)
            else:
                val = cycles_counter_conditional(all_actions_df, time_num_df, 5)

            mat[i,j] = val
            print(name, session)

    means = np.mean(mat, axis=1)
    x_values = np.arange(len(mat))

    
    fig, ax = plt.subplots()
    for i, row in enumerate(mat):
        for j, value in enumerate(row):
            ax.scatter(i+1, row[j], color='white', edgecolors='black', label = [name_list[j]] if i == 0 else "", alpha = 0.8)

    ax.plot(x_values +1, means, color='black', marker='o', linestyle='-')
    ax.set_xlabel('Session no.')
    ax.set_ylabel('No. of complete cycles (respecting the FR)')

    ax.set_xticks([1,2,3,4,5,6,7,8])
    ax.set_xticklabels(['FR1 Sess.1', 'FR1 Sess.2', 'FR1 Sess.3', 'FR1 Sess.4', 'FR5 Sess.1', 'FR5 Sess.2', 'FR5 Sess.3', 'FR5 Sess.4'], rotation=45)
    plt.tight_layout()

    #plt.legend()
    plt.show()

#plot_num_cycles_conditional()

def presses_per_seq():
    mat = np.zeros([8,len(name_list)])
    for j,name in enumerate(name_list):

            neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
            df = pd.read_csv(neurons_path)

            filtered_df = df[df.iloc[:,0].str.contains('FR1')]
            last_row_index = filtered_df.index[-1]

            new_list = []
            row_numbers = []

            for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
                new_list.append(elem)
                row_numbers.append(index)

            for i,session in enumerate(row_numbers):
                dat_path = df.iloc[session,0]
                data = load_data(dat_path)
                start_seq, end_seq = new_sequence(data)

                num_presses_list = []

                for (_, row1), (_, row2) in zip(start_seq.iterrows(), end_seq.iterrows()):
                    val = row2.iloc[7] - row1.iloc[7] + 1
                    num_presses_list.append(val)

                
                if len(num_presses_list) > 0:
                    mean = sum(num_presses_list) / len(num_presses_list) 
                else: 
                    mean = 0

                mat[i,j] = mean
                print(name, session)


    fig, ax = plt.subplots()
    for i, row in enumerate(mat):
        for j, value in enumerate(row):
            ax.scatter(i+1, row[j], label = [name_list[j]] if i == 0 else "", alpha = 0.8, color='white', edgecolors='black')

    ax.boxplot(mat.T, positions=np.arange(1, len(mat) + 1), showmeans=True, 
            medianprops=dict(color='black', linestyle='dotted'), 
            meanprops=dict(marker='^', markerfacecolor='red', markeredgewidth=0))


    plt.axhline(y=1, color='r', linestyle='--', xmin=0, xmax=0.5)
    plt.axhline(y=5, color='r', linestyle='--', xmin=0.5, xmax=1.0)

    ax.set_xlabel('Session no.')
    ax.set_ylabel('No. of presses per action sequence')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax.set_xticks([1,2,3,4,5,6,7,8])
    ax.set_xticklabels(['FR1 Sess.1', 'FR1 Sess.2', 'FR1 Sess.3', 'FR1 Sess.4', 'FR5 Sess.1', 'FR5 Sess.2', 'FR5 Sess.3', 'FR5 Sess.4'], rotation=45)
    plt.tight_layout()
    plt.show()

#presses_per_seq()





interval_list = []

for name in name_list: 


    neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
    df = pd.read_csv(neurons_path)

    filtered_df = df[df.iloc[:,0].str.contains('FR1')]
    last_row_index = filtered_df.index[-1]

    new_list = []
    row_numbers = []

    for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
        new_list.append(elem)
        row_numbers.append(index)

    #row_numbers = row_numbers[:4]
    row_numbers = row_numbers[4:]

    
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
                    if num >= 5:
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

        print(result_df)

        if len(result_df) > 0:
            timing_list = result_df[result_df.iloc[:,1] == 'sequence'].iloc[:,0].tolist()

            print(timing_list)


            for i in range(len(timing_list) -1):
                        interval = timing_list[i+1] - timing_list[i]
                        print(interval)
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

plt.xlabel('Duration of complete cycles of FR5 sessions (ms)')
plt.ylabel('Frequency')

plt.xlim(10000, 130000)
plt.tight_layout()
plt.show()

