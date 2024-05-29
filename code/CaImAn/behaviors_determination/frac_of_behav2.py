import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys

sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from behaviors_function import *

def binary_classes2(row):

    list = row.tolist()
    output = [0] * 8

    for i,element in enumerate(list):
        if i == 0:
            if element >= 95 : 
                output[0] = 1
            if element <= 5 : 
                output[4] = 1
        if i == 1 or i == 2:
            if element >= 95 : 
                output[1] = 1
            if element <= 5 : 
                output[5] = 1
        if i == 3 or i == 4:
            if element >= 95 : 
                output[2] = 1
            if element <= 5 : 
                output[6] = 1
        if i == 5 or i == 6:
            if element >= 95 : 
                output[3] = 1
            if element <= 5 : 
                output[7] = 1               
    return output

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


folder = 'neurons'

name_list = ['I2', 'I3', 'I6']
name_list = ['L0', 'L2', 'L3']


if name_list == ['L0', 'L2', 'L3']:
    color = 'darkblue'
if name_list == ['I2', 'I3', 'I6']:
    color = 'brown'


def get_num_neurons():
    num_active_mat = np.zeros((3,8))
    num_inactive_mat = np.zeros((3,8))
    num_nonsig_mat = np.zeros((3,8))

    for i,name in enumerate(name_list):

        row_numbers = extract_sessions(name)
        for j,session in enumerate(row_numbers):

            print(name,session)

            percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session+1}.csv'
            percentile_df = pd.read_csv(percentile_path,header=None)

            binary_rows_list = []
            for index, row in percentile_df.iterrows():
                binary_row = binary_classes2(row)
                binary_rows_list.append(binary_row)


            binary_df = pd.DataFrame(binary_rows_list)
            means = (binary_df.sum()).tolist()

            active_sub_df = binary_df.iloc[:, :4]
            active_sub_df = active_sub_df[(active_sub_df != 0).any(axis=1)]
            num_active_neurons = active_sub_df.shape[0]

            inactive_sub_df = binary_df.iloc[:, 4:]
            inactive_sub_df = inactive_sub_df[(inactive_sub_df != 0).any(axis=1)]
            num_inactive_neurons = inactive_sub_df.shape[0]

            nonsig_df = binary_df[binary_df.eq(0).all(axis=1)]
            num_nonsig = nonsig_df.shape[0]

            num_active_mat[i,j] = num_active_neurons/ len(percentile_df)
            num_inactive_mat[i,j] = num_inactive_neurons/ len(percentile_df)

            num_nonsig_mat[i,j] = num_nonsig / len(percentile_df)


    num_active_df = pd.DataFrame(num_active_mat)
    num_inactive_df = pd.DataFrame(num_inactive_mat)
    num_nonsig_df = pd.DataFrame(num_nonsig_mat)

    return num_active_df, num_inactive_df, num_nonsig_df

def plot_num_neurons():

    num_active_df, num_inactive_df, num_nonsig_df = get_num_neurons()


    mean_active_list = num_active_df.mean()
    mean_inactive_list = num_inactive_df.mean()

    mean_nonsig_list = num_nonsig_df.mean()

    color_list = ['darksalmon', 'lightcoral', 'indianred', 'brown', 'skyblue', 'cornflowerblue', 'royalblue', 'blue']
    plt.figure(figsize=(9,5))

    
    plt.plot(np.arange(0,8), mean_active_list, color=color, linewidth=3,label='Significantly active')
    plt.plot(np.arange(0,8), mean_inactive_list, color=color, linewidth=3, linestyle='dotted', label='Significantly inactive')

    plt.plot(np.arange(0,8), mean_nonsig_list, color='grey',alpha=0.6, linewidth=3, linestyle='--', label='Nonsignificant')



    # conf_int_active = np.percentile(num_active_df, [2.5, 97.5], axis=0)
    # plt.fill_between(np.arange(0, 8), conf_int_active[0], conf_int_active[1], alpha=0.2, color=color)

    # conf_int_inactive = np.percentile(num_inactive_df, [2.5, 97.5], axis=0)
    # plt.fill_between(np.arange(0, 8), conf_int_inactive[0], conf_int_inactive[1], alpha=0.2, color=color, hatch ='.')

    # conf_int_nonsig = np.percentile(num_nonsig_df, [2.5, 97.5], axis=0)
    # plt.fill_between(np.arange(0, 8), conf_int_nonsig[0], conf_int_nonsig[1], alpha=0.2, color='green', hatch ='.')

    plt.ylabel('Percentage of detected neurons')
    plt.xticks([])

    #plt.legend(loc=(0.7,0.3)) 
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.xlim(-0.3,7.3)
    plt.ylim(0,0.7)
    plt.show()

#plot_num_neurons()


def get_freq_behav():

    freq_active_mat = []
    freq_inactive_mat = []
    for i,name in enumerate(name_list):

        row_active = []
        row_inactive = []

        row_numbers = extract_sessions(name)
        for j,session in enumerate(row_numbers):

            print(name,session)

            percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session+1}.csv'
            percentile_df = pd.read_csv(percentile_path,header=None)

            binary_rows_list = []
            for index, row in percentile_df.iterrows():
                binary_row = binary_classes2(row)
                binary_rows_list.append(binary_row)


            binary_df = pd.DataFrame(binary_rows_list)
            means = (binary_df.sum()).tolist()



            active_sub_df = binary_df.iloc[:, :4]
            active_sub_df = active_sub_df[(active_sub_df != 0).any(axis=1)]
            num_active_neurons = active_sub_df.shape[0]

            inactive_sub_df = binary_df.iloc[:, 4:]
            inactive_sub_df = inactive_sub_df[(inactive_sub_df != 0).any(axis=1)]
            num_inactive_neurons = inactive_sub_df.shape[0]


            means = [(x / num_active_neurons) if i < 4 else (x / num_inactive_neurons) for i, x in enumerate(means)]

            row_active.extend(means[:4])
            row_inactive.extend(means[4:])
        
        freq_active_mat.append(row_active)
        freq_inactive_mat.append(row_inactive)

    freq_active_df = pd.DataFrame(freq_active_mat)
    freq_inactive_df = pd.DataFrame(freq_inactive_mat)     


    #####

    ind_list = []
    for i in range(4):
        for j in range(8):
            ind_list.append(i +4*j)

    freq_active_df = freq_active_df.iloc[:,ind_list]
    freq_inactive_df = freq_inactive_df.iloc[:,ind_list]


    pos = []
    for i in range(4):
        for j in range(8):
        #sub = [0+i*(num_clusters+1), 1+i*(num_clusters+1)]
            pos.append(j+i*(8+1))


    color_list = ['darksalmon', 'lightcoral', 'indianred', 'brown', 'skyblue', 'cornflowerblue', 'royalblue', 'blue']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(freq_active_df.shape[1]):
        col = freq_active_df.iloc[:, i]
        col_mean = col.mean(skipna=True)
        ax.bar(pos[i], col_mean, color = color_list[i%8], edgecolor = 'black', label = f' FR{((i // 4) * 4) + 1} Sess. {(i % 4) + 1}')
        ax.scatter([pos[i]] * len(col), col, color='none', edgecolor='black', s=14)

    #plt.xticks(pos, [f'Sess. {p%8 +1}' for p in range(freq_active_df.shape[1])], rotation='vertical')
    #plt.legend()
    plt.ylabel('Percentage of significantly active neurons')
    plt.xticks([])
    plt.xlim(-1,35)
    plt.tight_layout()
    plt.show()



    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(freq_inactive_df.shape[1]):
        col = freq_inactive_df.iloc[:, i]
        col_mean = col.mean(skipna=True)
        ax.bar(pos[i], col_mean, color = color_list[i%8], edgecolor = 'black', label = f'Sess. {i+1}')
        ax.scatter([pos[i]] * len(col), col, color='none', edgecolor='black', s=14)

    #plt.xticks(pos, [f'Sess. {p%8 +1}' for p in range(freq_active_df.shape[1])], rotation='vertical')
    #plt.legend()
    plt.ylabel('Percentage of significantly inactive neurons')
    plt.xticks([])
    plt.xlim(-1,35)
    plt.tight_layout()
    plt.show()


#get_freq_behav()

def get_nb():
    nb_active_mat = []
    nb_inactive_mat = []
    for i,name in enumerate(name_list):

        row_active = []
        row_inactive = []

        row_numbers = extract_sessions(name)
        for j,session in enumerate(row_numbers):

            print(name,session)


            percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session+1}.csv'
            percentile_df = pd.read_csv(percentile_path,header=None)

            binary_rows_list = []
            for index, row in percentile_df.iterrows():
                binary_row = binary_classes2(row)
                binary_rows_list.append(binary_row)


            binary_df = pd.DataFrame(binary_rows_list)
            means = (binary_df.sum()).tolist()

            active_sub_df = binary_df.iloc[:, :4]
            inactive_sub_df = binary_df.iloc[:, 4:]

            sum_active_list = active_sub_df.sum(axis=1).tolist()
            count_active_list = [sum_active_list.count(i) for i in range(5)]

            sum_inactive_list = inactive_sub_df.sum(axis=1).tolist()
            count_inactive_list = [sum_inactive_list.count(i) for i in range(5)]

            count_active_list = count_active_list / np.sum(count_active_list)
            count_inactive_list = count_inactive_list / np.sum(count_inactive_list)
            
            row_active.extend(count_active_list)
            row_inactive.extend(count_inactive_list)

        nb_active_mat.append(row_active)
        nb_inactive_mat.append(row_inactive)

    nb_active_df = pd.DataFrame(nb_active_mat)
    nb_inactive_df = pd.DataFrame(nb_inactive_mat)




    pos = []
    for i in range(8):
        pos.extend([0+6*i, 1+6*i, 2+6*i, 3+6*i, 4+6*i])
    print(pos)

    print(len(pos))
    print(nb_active_df.shape[1])

    color_list = ['indianred', 'skyblue', 'cornflowerblue', 'royalblue', 'blue']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(nb_active_df.shape[1]):
        col = nb_active_df.iloc[:, i]
        col_mean = col.mean(skipna=True)
        ax.bar(pos[i], col_mean, edgecolor = 'black', color = color_list[i%5])
        ax.scatter([pos[i]] * len(col), col, color='none', edgecolor='black', s=14)

    plt.show()



    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(nb_inactive_df.shape[1]):
        col = nb_inactive_df.iloc[:, i]
        col_mean = col.mean(skipna=True)
        ax.bar(pos[i], col_mean, edgecolor = 'black', color = color_list[i%5])
        ax.scatter([pos[i]] * len(col), col, color='none', edgecolor='black', s=14)

    plt.show()