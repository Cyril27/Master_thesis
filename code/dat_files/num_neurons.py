import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind




def get_lr_path(name):

    list = []

    list.append(['L0', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L0/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['L2', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L2/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['L3', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L3/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['L4', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L4/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['P4','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_a2ap4/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['D1_1','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/1 triss/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['D1_2','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/2 phi/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['D1_3','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/3 tis/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['I2','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i2/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['I3','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i3/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['I6','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i6/Extracted_Data/Longitudinal_Registration.csv'])

    lr_path_df = pd.DataFrame(list)
    long_reg_path = lr_path_df[lr_path_df[0] == name].iloc[0,1]

    return long_reg_path

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


#name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6', 'D1_1', 'D1_2', 'D1_3', 'P4']
name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']


def neurons_d_i():

    direct_data = []
    indirect_data = []

    for name in name_list:

        row_numbers = extract_sessions(name)

        long_reg_path = get_lr_path(name)
        long_reg_df = pd.read_csv(long_reg_path, header=None)


        if name == 'I6':
            long_reg_df = long_reg_df.drop(long_reg_df.columns[2], axis=1)
            long_reg_df.columns = range(len(long_reg_df.columns))


        for sess_id, session in enumerate(row_numbers):
            max_val = max(long_reg_df.iloc[:,session])

            if name in ['I2', 'I3', 'I6','D1_1', 'D1_2', 'D1_3']:
                #plt.scatter(0, max_val, color = 'white', edgecolors='black')
                direct_data.append(max_val)
            if name in ['L0', 'L2', 'L3']:
                #plt.scatter(1, max_val, color = 'white', edgecolors='black')
                indirect_data.append(max_val)

    plt.xticks([0, 1],['Direct', 'Indirect'])

    box = plt.boxplot([direct_data, indirect_data],
                    widths=0.6,
                    labels=['Direct', 'Indirect'],
                    positions=[0, 1],
                    showmeans=True,
                    patch_artist=True,  
                    medianprops=dict(color='black', linestyle='dotted'),  
                    meanprops=dict(marker='^', markerfacecolor='black', markeredgewidth=0))  


   
    colors = ['brown', 'darkblue']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.66)

    plt.scatter([0] * len(direct_data), direct_data, color='white', edgecolors='black', zorder=2)
    plt.scatter([1] * len(indirect_data), indirect_data, color='white', edgecolors='black', zorder=2)


    statistic, p_value = mannwhitneyu(direct_data, indirect_data)
    print("Mann Withney P-value:", p_value)

    t_statistic, p_value = ttest_ind(direct_data, indirect_data)
    print("t-test P-value:", p_value)

    plt.xlabel('Pathway')
    plt.ylabel('Number of neruons detected')
    plt.tight_layout()
    plt.show()

#neurons_d_i()


def neurons_d_i_per_session():

    list_direct = ['I2', 'I3', 'I6']
    list_indirect = ['L0', 'L3']

    matrix_direct = np.zeros((8,len(list_direct)))
    matrix_indirect = np.zeros((8,len(list_indirect)))

    #fig, axes = plt.subplots(2, 4, figsize=(13, 5))  # Create a 2x4 subplot grid

    for name in name_list:

        row_numbers = extract_sessions(name)

        long_reg_path = get_lr_path(name)
        long_reg_df = pd.read_csv(long_reg_path, header=None)


        if name == 'I6':
            long_reg_df = long_reg_df.drop(long_reg_df.columns[2], axis=1)
            long_reg_df.columns = range(len(long_reg_df.columns))


        for sess_id, session in enumerate(row_numbers):
            max_val = max(long_reg_df.iloc[:,session])

            #ax = axes[sess_id // 4, sess_id % 4]  

            if name in list_direct:
                #ax.scatter(0, max_val, color = 'white', edgecolors='black')
                matrix_direct[sess_id,list_direct.index(name)] = max_val
                
            if name in list_indirect:
                #ax.scatter(1, max_val, color = 'white', edgecolors='black')
                matrix_indirect[sess_id,list_indirect.index(name)] = max_val

            #ax.set_title(f'Session {sess_id+1}') 
            #ax.set_xticks([0, 1],['Direct', 'Indirect'])

            
    max_value = max(matrix_direct.max(), matrix_indirect.max())

    # for  i in range(8):
    #     sub_direct = matrix_direct[i,:]
    #     sub_indirect = matrix_indirect[i,:]

    #     # means_direct.append(np.mean(sub_direct))
    #     # means_indirect.append(np.mean(sub_indirect))

    
    #     ax = axes[i // 4, i % 4]  
    #     ax.boxplot([sub_direct, sub_indirect], widths=0.6, labels=['Direct', 'Indirect'], positions =[0,1])


    #     ax.set_ylim(0, max_value)

    # plt.tight_layout()
    #plt.show()


    # plt.figure()
    # plt.plot(np.arange(1,9), means_direct, label='direct')
    # plt.plot(np.arange(1,9), means_indirect, label='indirect')

    # sum_of_means = [mean_direct + mean_indirect for mean_direct, mean_indirect in zip(means_direct, means_indirect)]
    # plt.plot(np.arange(1,9), sum_of_means, label='Sum of Means')

    # plt.legend()
    # #plt.show()


    means_direct = np.mean(matrix_direct, axis=1)
    conf_int_direct = np.percentile(matrix_direct, [2.5, 97.5], axis=1)

    
    means_indirect = np.mean(matrix_indirect, axis=1)
    conf_int_indirect = np.percentile(matrix_indirect, [2.5, 97.5], axis=1)

    sum_of_means = [mean_direct + mean_indirect for mean_direct, mean_indirect in zip(means_direct, means_indirect)]


    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(1, 9), y=means_direct, marker='o', label='Direct pathway', color='brown')
    plt.fill_between(np.arange(1, 9), conf_int_direct[0], conf_int_direct[1], alpha=0.3, color='red')
    sns.lineplot(x=np.arange(1, 9), y=means_indirect, marker='o', label='Indirect pathway', color='darkblue')
    plt.fill_between(np.arange(1, 9), conf_int_indirect[0], conf_int_indirect[1], alpha=0.3, color='blue')

    #sns.lineplot(x=np.arange(1, 9), y=sum_of_means, marker='o', label='Sums of means', color='black')




    #plt.xlabel('Session No.')
    plt.xticks([])
    plt.ylabel('No. of neurons detected')
    
    plt.legend(loc='upper left')
    plt.show()

#neurons_d_i_per_session()


#name_list = ['I2', 'I3', 'I6']
name_list = ['L0', 'L2', 'L3']


def following_neurons():

    matrix = np.zeros((8, 8))
    for name in name_list:

        row_numbers = extract_sessions(name)
        long_reg_path = get_lr_path(name)
        long_reg_df = pd.read_csv(long_reg_path, header=None)

        if name == 'I6':
            long_reg_df = long_reg_df.drop(long_reg_df.columns[2], axis=1)
            long_reg_df.columns = range(len(long_reg_df.columns))


        reduced_df = long_reg_df.iloc[:,row_numbers]
        reduced_df.columns = range(len(reduced_df.columns))


        for idx, row in reduced_df.iterrows():
            row_list = row.tolist()
            for col1 in range(8):
                for col2 in range(8):
                    if row_list[col1] != 0 and row_list[col2] != 0:
                        matrix[col1,col2] += 1
                

        # row_max_values = matrix.max(axis=1, keepdims=True)
        # matrix = matrix / row_max_values


    min_values = np.zeros_like(matrix)

    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            index = min(i,j)
            min_values[i,j] = matrix[i,j]/matrix[index,index]



    sns.heatmap(min_values, cmap='Reds', annot=True, fmt=".2f", vmin=0.56, vmax=1.0)
    #plt.xticks(np.arange(matrix.shape[1]) + 0.5, np.arange(matrix.shape[1]) + 1)
    #plt.yticks(np.arange(matrix.shape[0]) + 0.5, np.arange(matrix.shape[0]) + 1)
    plt.xticks([])
    plt.yticks([])
    plt.show()


following_neurons()