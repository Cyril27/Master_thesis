import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from behaviors_function import *
from image_function import *

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


def get_means_of_behavs():
    for name in name_list:

        big_row = []

        neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
        df = pd.read_csv(neurons_path)

        row_numbers = extract_sessions(name)

        for session in row_numbers:

            print(name,session)


            seq_list = []
            zone1_list = []
            zone2_list = []
            pellet_list = []
            nopellet_list = []
            explo1_list = []
            explo2_list = []


            dat_path = df.iloc[session,0]
            all_actions_df = number_cycles(dat_path)



            for i in range(len(all_actions_df)-2):
                behav = all_actions_df.iloc[i,1]
                duration = all_actions_df.iloc[i+1,0] - all_actions_df.iloc[i,0]
                
                if behav == 'sequence':
                    seq_list.append(duration)
                if behav == 'zone 1':
                    zone1_list.append(duration)
                if behav == 'zone 2':
                    zone2_list.append(duration)
                if behav == 'pellet':
                    pellet_list.append(duration)
                if behav == 'no pellet':
                    nopellet_list.append(duration)
                if behav == 'explo 1':
                    explo1_list.append(duration)
                if behav == 'explo 2':
                    explo2_list.append(duration)

            row = [np.mean(seq_list), np.mean(zone1_list), np.mean(zone2_list), np.mean(pellet_list), np.mean(nopellet_list), np.mean(explo1_list), np.mean(explo2_list)]
            row = [0.0 if np.isnan(x) else x for x in row]

            big_row.extend(row)

        result_mat.append(big_row)

    mean_df = pd.DataFrame(result_mat)
    #mean_df.to_csv('/Users/cyrilvanleer/Desktop/Thesis/dat_files/means_behav.csv', header=False, index=False)


def plot_duration_behavior():
    mean_df = pd.read_csv('/Users/cyrilvanleer/Desktop/Thesis/dat_files/means_behav.csv', header=None)
    means = mean_df.mean()


    for i in range(8):
        sub_means = means.iloc[i*7: (i+1)*7]
        sub_complete = sub_means.iloc[:4].tolist()
        print(sub_complete)
        print(np.sum(sub_means))



    mult = 9
    pos_list = []
    for i in range(8):
        pos_list.extend([0+i*mult, 1+i*mult, 2+i*mult, 3+i*mult, 4+i*mult, 5+i*mult, 6+i*mult])

    ticks_list = []
    label_list = ['Sequence', 'Zone 1', 'Zone 2', 'Pellet', 'No pellet', 'Explo 1', 'Explo 2']
    for i in range(8):
        ticks_list.extend(label_list)

    color_list = ['blue', 'green', 'red', 'purple', 'orange', 'lightgray', 'dimgray']

    plt.figure(figsize=(12,5))
    for i, (pos, mean) in enumerate(zip(pos_list, means)):
        color = color_list[i % len(color_list)]  
        plt.bar(pos, mean, color=color)
        
        plt.scatter([pos] * 6, mean_df.iloc[:,i], color='none', edgecolors='black', s=14)

    #plt.xticks(pos_list, ticks_list, rotation='vertical') 
    plt.ylabel('Time (ms)')
    plt.tight_layout()
    plt.ylim(0,26000)
    plt.xlim(-1, 70)

    plt.xticks([])
    plt.show()


#plot_duration_behavior()


def get_means_of_behavs_complete():

    for name in name_list:

        big_row = []

        neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
        df = pd.read_csv(neurons_path)

        row_numbers = extract_sessions(name)

        for sess_id,session in enumerate(row_numbers):

            print(name,session)

            if sess_id <= 3:
                cond = 1
            if sess_id > 3:
                cond = 5


            seq_list = []
            zone1_list = []
            zone2_list = []
            pellet_list = []
            nopellet_list = []
            explo1_list = []
            explo2_list = []


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
                        if num >= cond:
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
            #print(result_df)




            for i in range(len(result_df)-2):
                behav = result_df.iloc[i,1]

                sub_end = all_actions_df[all_actions_df.iloc[:,0] > result_df.iloc[i,0]]
                end_val = sub_end.iloc[0,0]
                
                duration = result_df.iloc[i+1,0] - result_df.iloc[i,0]
                
                if behav == 'sequence':
                    seq_list.append(duration)
                if behav == 'zone 1':
                    zone1_list.append(duration)
                if behav == 'zone 2':
                    zone2_list.append(duration)
                if behav == 'pellet':
                    pellet_list.append(duration)


            row = [np.mean(seq_list), np.mean(zone1_list), np.mean(zone2_list), np.mean(pellet_list)]
            row = [0.0 if np.isnan(x) else x for x in row]


            big_row.extend(row)

        result_mat.append(big_row)

    mean_df = pd.DataFrame(result_mat)

    print(mean_df)


    means = mean_df.mean()


    for i in range(8):
        sub_means = means.iloc[i*4: (i+1)*4]
        #sub_complete = sub_means.iloc[:4].tolist()
        print(sub_means)
        print(np.sum(sub_means))



    mult = 8.5
    pos_list = []
    for i in range(8):
        pos_list.extend([0+i*mult, 1+i*mult, 2+i*mult, 3+i*mult,])

    ticks_list = []
    label_list = ['Sequence', 'Zone 1', 'Zone 2', 'Pellet']
    for i in range(8):
        ticks_list.extend(label_list)

    color_list = ['blue', 'green', 'red', 'purple']

    plt.figure(figsize=(12,5))
    for i, (pos, mean) in enumerate(zip(pos_list, means)):
        color = color_list[i % len(color_list)]  
        plt.bar(pos, mean, color=color)
        
        plt.scatter([pos] * 6, mean_df.iloc[:,i], color='none', edgecolors='black', s=14)

    #plt.xticks(pos_list, ticks_list, rotation='vertical') 
    plt.xticks([])
    plt.ylabel('Time (ms)')
    plt.tight_layout()
    #plt.ylim(0,26000)
    plt.xlim(-1, 70)
    plt.show()

get_means_of_behavs_complete()