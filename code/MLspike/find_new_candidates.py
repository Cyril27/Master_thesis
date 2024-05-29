import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
from scipy.stats import percentileofscore

sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from behaviors_function import *


big_name = 'L023'
name = 'L3'


csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
df = pd.read_csv(csv_file_path)


neighbour_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/{name}/combined_binary.csv'
neighbour_df = pd.read_csv(neighbour_path, header=None)

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



long_reg_path = get_lr_path(name)
long_reg_df = pd.read_csv(long_reg_path, header=None)


if name == 'I6':
    long_reg_df = long_reg_df.drop(2, axis=1) 
    new_column_names = range(len(long_reg_df.columns))
    long_reg_df.columns = new_column_names  


if name == 'L0' or name == 'L3' or name == 'I2':
    long_reg_df = long_reg_df.iloc[:100,:]
if name == 'L2':
    long_reg_df = long_reg_df.iloc[:50,:]
if name == 'D1_1':
    long_reg_df = long_reg_df.iloc[:57,:]
if name == 'D1_3':
    long_reg_df = long_reg_df.iloc[:44,:]
if name == 'I3':
    long_reg_df = long_reg_df.iloc[:107,:]
if name == 'I6':
    long_reg_df = long_reg_df.iloc[:101,:]
if name == 'P4':
    long_reg_df = long_reg_df.iloc[:117,:]

row_numbers = extract_sessions(name)

for session_id, session in enumerate(row_numbers):
    print('####' *5)
    print(session_id)


    percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/percentiles/{name}/S{session+1}.csv'
    percentile_df = pd.read_csv(percentile_path, header=None)

    remove_index = long_reg_df.iloc[:,session].tolist()
    remove_index = [value-1 for value in remove_index if value != 0]


    percentile_df = percentile_df.drop(index=remove_index)
    print(percentile_df)
    index_list = percentile_df.index.tolist()


    binary_rows_list = []
    for index, row in percentile_df.iterrows():
        binary_row = binary_classes2(row)
        binary_rows_list.append(binary_row)


    binary_df = pd.DataFrame(binary_rows_list)


    session_neighbour_df = neighbour_df.iloc[:,8*session_id:8*(session_id+1)]

    

    if big_name == 'I236':
        cluster_path = '/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/I236/neuron_class_mouse_6_average.csv'

    if big_name == 'L023':
        cluster_path = '/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/L023/neuron_class_mouse_6.csv'
    
    cluster_df = pd.read_csv(cluster_path, header=None)

    



    #### For all names

    if big_name == 'I236':
        name_list = ['I2', 'I3', 'I6']
    if big_name == 'L023':
        name_list = ['L0', 'L2', 'L3']
    
    centroid_list = []
    for i in range(1,7):
        big_centroid = []
        cluster_i = cluster_df[cluster_df.iloc[:,1] == i]

        for sub_name in name_list:
            cluster_name = cluster_i[cluster_i.iloc[:,2] == sub_name]
            long_reg_index = cluster_name.iloc[:,0].tolist()


            if sub_name == 'I3':
                long_reg_index = [elem - 100 for elem in long_reg_index]
            if sub_name == 'I6':
                long_reg_index = [elem - 207 for elem in long_reg_index]

            if sub_name == 'L2':
                long_reg_index = [elem - 100 for elem in long_reg_index]
            if sub_name == 'L3':
                long_reg_index = [elem - 150 for elem in long_reg_index]

            neighbour_i_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/{sub_name}/combined_binary.csv'
            neighbour_i_df = pd.read_csv(neighbour_i_path, header=None)

            neighbour_i_df = neighbour_i_df.iloc[:,8*session_id:8*(session_id+1)]

            
            cluster_i_binary = neighbour_i_df.iloc[long_reg_index,:]

            for id,row in cluster_i_binary.iterrows():
                list = row.tolist()
                #big_centroid.append(list)

        big_centroid_df = pd.DataFrame(big_centroid)

        centroid = cluster_i_binary.mean(axis=0).tolist()
        centroid_list.append(centroid)



    ##### for just the name
    
    # cluster_df = cluster_df[cluster_df.iloc[:,2] == name]
    # print(cluster_df)

    # centroid_list = []
    # for i in range(1,7):
    #     cluster_i = cluster_df[cluster_df.iloc[:,1] == i]
    #     long_reg_index = cluster_i.iloc[:,0].tolist()

    #     if name == 'I3':
    #         long_reg_index = [elem - 100 for elem in long_reg_index]
    #     if name == 'I6':
    #         long_reg_index = [elem - 207 for elem in long_reg_index]

    #     if name == 'L2':
    #         long_reg_index = [elem - 100 for elem in long_reg_index]
    #     if name == 'L3':
    #         long_reg_index = [elem - 150 for elem in long_reg_index]


    #     cluster_i_binary = session_neighbour_df.iloc[long_reg_index,:]
    #     centroid = cluster_i_binary.mean(axis=0).tolist()
    #     centroid_list.append(centroid)






    centroid_df = pd.DataFrame(centroid_list)


    mat = np.zeros((len(binary_df), len(centroid_df)))

    for i,row in binary_df.iterrows():
        for j, centroid in centroid_df.iterrows():
            euclidean_distance = np.linalg.norm(row - centroid)
            mat[i,j] = euclidean_distance


    distance_df = pd.DataFrame(mat)


    list = []
    for idx, row in distance_df.iterrows():
        cluster = row.idxmin() +1 
        print(idx, cluster)

        list.append([idx,index_list[idx], cluster, binary_df.iloc[idx,:].tolist()])

    
    output_df = pd.DataFrame(list)
    print(output_df)

    output_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/more_neurons/{name}_all/S{session+1}.csv', index=False, header=False)



