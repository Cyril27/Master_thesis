import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

big_name = 'L023'

if big_name == 'I236':
    name_list = ['I2', 'I3', 'I6']
if big_name == 'L023':
    name_list = ['L0', 'L2', 'L3']


all_list = ['I2', 'I3', 'I6', 'L0', 'L2', 'L3']

color_list = ['blue', 'red', 'green']


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

def plot_last_centroids():

    fig, axes = plt.subplots(2, 3, figsize=(13, 5)) 
    for i,name in enumerate(all_list):

        row_numbers = extract_sessions(name)
        session = row_numbers[-1]
        
        centroid_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/centroids/{name}/S{session+1}.csv'
        centroid_df = pd.read_csv(centroid_path, header=None)

        ax = axes[i // 3, i % 3]
        ax.scatter(centroid_df[0], centroid_df[1], label=name, color = 'white', edgecolors='black')
        ax.set_title(name)  

    plt.tight_layout()
    plt.show()

plot_last_centroids()

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


def plot_centroid_cluster():

    fig, axes = plt.subplots(1, len(name_list), figsize=(15, 4)) 

    for name_id,name in enumerate(name_list):
        row_numbers = extract_sessions(name)
        session = row_numbers[-1]

        print(session)

        centroid_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/centroids/{name}/S{session+1}.csv'
        centroid_df = pd.read_csv(centroid_path, header=None)

        data_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/similarities_8/{big_name}/data_stat.csv'
        data_df = pd.read_csv(data_path)

        sub_df = data_df.drop_duplicates(subset='Neuron_ID')
        sub_df = sub_df[ sub_df['Name'] == name]
        print(sub_df)
        print(len(sub_df))

        long_reg_path = get_lr_path(name)
        long_reg_df = pd.read_csv(long_reg_path, header=None)


        centroid_cluster_list = []

        for id,row in sub_df.iterrows():
            
            long_reg_index = row['Neuron_ID']
            if name == 'I3':
                long_reg_index -= 100
            if name == 'I6':
                long_reg_index -= 207
            if name == 'L2':
                long_reg_index -= 100
            if name == 'L3':
                long_reg_index -= 150


            cluster = row['Cluster']

            session_id = long_reg_df.iloc[long_reg_index, session]

            new_row = [centroid_df.iloc[session_id-1,0], centroid_df.iloc[session_id-1,1], cluster]

            centroid_cluster_list.append(new_row)


        centroid_cluster_df = pd.DataFrame(centroid_cluster_list)
        #print(centroid_cluster_df)


        #colors = {'Cluster 1': 'red', 'Cluster 2': 'blue', 'Cluster 3': 'green', 'Cluster 4': 'purple', 'Cluster 5': 'gray', 'Cluster 6': 'orange'}
        
        #colors = {'Cluster 1': 'red', 'Cluster 2': 'blue', 'Cluster 3': 'blue', 'Cluster 4': 'purple', 'Cluster 5': 'orange', 'Cluster 6': 'orange'}

        colors = {'Cluster 1': 'red', 'Cluster 2': 'blue', 'Cluster 3': 'green', 'Cluster 4': 'green', 'Cluster 5': 'green', 'Cluster 6': 'green'}


        for i in range(len(centroid_cluster_df)):
            axes[name_id].scatter(centroid_cluster_df.iloc[i, 0], centroid_cluster_df.iloc[i, 1], color=colors[centroid_cluster_df.iloc[i, 2]])

        axes[name_id].set_xlim(0, 320)  
        axes[name_id].set_ylim(0, 200)  
        axes[name_id].set_title(name) 

    plt.tight_layout()
    plt.show()


#plot_centroid_cluster()


# name = 'L3'

# plt.figure()
# centroid_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/centroids/{name}/S8.csv'
# centroid_df = pd.read_csv(centroid_path, header=None)


# plt.scatter(centroid_df[0], centroid_df[1])
# plt.show()






def plot_all_sessions(name):

    row_sessions = extract_sessions(name)
    
    fig, axes = plt.subplots(2, 4, figsize=(13, 5))  
    
    for i, session in enumerate(row_sessions):
        centroid_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/centroids/{name}/S{session+1}.csv'
        centroid_df = pd.read_csv(centroid_path, header=None)

        ax = axes[i // 4, i % 4]
        ax.scatter(centroid_df[0], centroid_df[1], label=f'Session {session+1}')
        ax.set_title(f'Session {session+1}')  

    # Adjust layout
    plt.tight_layout()
    #plt.show()




#plot_all_sessions('L2')


def plot_heatmaps(name):

    row_sessions = extract_sessions(name)
    
    fig, axes = plt.subplots(2, 4, figsize=(13, 5))  # Create a 2x4 subplot grid
    
    for i, session in enumerate(row_sessions):
        centroid_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/centroids/{name}/S{session+1}.csv'
        centroid_df = pd.read_csv(centroid_path, header=None)
        
        ax = axes[i // 4, i % 4]  
        
        sns.kdeplot(x=centroid_df[0], y=centroid_df[1], cmap="coolwarm", ax=ax, fill=True, levels=100)
        ax.set_title(f'Session {session+1}') 

    # Adjust layout
    plt.tight_layout()
    plt.show()


#plot_heatmaps('L2')