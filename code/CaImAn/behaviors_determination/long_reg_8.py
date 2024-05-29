import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import jaccard_score



pd.set_option('display.max_colwidth', None)



folder = 'deconvoled'


save_csv = True
name = 'L3'

percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}'
neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'

print(percentile_path)

df = pd.read_csv(neurons_path)

filtered_df = df[df.iloc[:,0].str.contains('FR1')]
last_row_index = filtered_df.index[-1]

new_list = []
row_numbers = []

for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
    new_list.append(elem)
    row_numbers.append(index+1)

dfs = {f'df{index}': pd.read_csv(f'{percentile_path}/S{val}.csv', header=None) for index,val in enumerate(row_numbers)}

print(row_numbers)

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


binary_dfs = {}
for i in range(8):
    binary_rows_list = []

    df = list(dfs.values())[i]
    #df = df.iloc[:2,:]

    for index, row in df.iterrows():
        binary_row = binary_classes2(row)
        binary_rows_list.append(binary_row)

    binary_dfs[f'df{i}'] = pd.DataFrame(binary_rows_list)



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

long_reg_path = get_lr_path(name)
long_reg_df = pd.read_csv(long_reg_path, header=None)

if name == 'D1_2':
    long_reg_df = long_reg_df.drop(10, axis=1)             # ONLY FOR D1_2
if name == 'I6':
    long_reg_df = long_reg_df.drop(2, axis=1)               # ONLY FOR I6



columns_to_keep = [x - 1 for x in row_numbers]
long_reg_df = long_reg_df.iloc[:, columns_to_keep]



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


#L0: 100 
#L2: 50
#L3: 100
#D1_1: 57
#D1_3: 44
#I2: 100
#I3: 107
#I6: 101
#P4: 117

def combined_binary():
    combined_all = []

    for index,row in long_reg_df.iterrows():
        combined_list = []

        for idx, elem in enumerate(row.tolist()):
            df = list(binary_dfs.values())[idx]
            bin_list = df.iloc[elem-1,:]
            combined_list.extend(bin_list)

        combined_all.append(combined_list)

    df = pd.DataFrame(combined_all)
    return df


df = combined_binary()

if save_csv:
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/combined_binary.csv'
    df.to_csv(output_file, index=False, header=False)
    

def combined_percentile():
    combined_all = []

    for index,row in long_reg_df.iterrows():
        combined_list = []

        for idx, elem in enumerate(row.tolist()):
            df = list(dfs.values())[idx]
            bin_list = df.iloc[elem-1,:]
            combined_list.extend(bin_list)

        combined_all.append(combined_list)

    df = pd.DataFrame(combined_all)
    return df


percentile_df = combined_percentile()
if save_csv:
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/combined_percentile.csv'
    percentile_df.to_csv(output_file, index=False, header=False)







def combined_similarity():

    df = combined_binary()
    result_df = pd.DataFrame(index=df.index, columns=df.index)

    for index1,row1 in df.iterrows():
        for index2,row2 in df.iterrows():

            sim = jaccard_score(row1.tolist(), row2.tolist())
            result_df.loc[index1, index2] = sim      

    result_df = result_df.astype(float)
    return result_df

mat = combined_similarity()
if save_csv:
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/all.csv'
    mat.to_csv(output_file, index=False, header=False)



def verification_last2_FR5():

    df = combined_binary()
    result_df = pd.DataFrame(index=df.index, columns=df.index)

    for index1,row1 in df.iterrows():
        for index2,row2 in df.iterrows():
            row1_reduced = row1[-16:]
            row2_reduced = row2[-16:]

            sim = jaccard_score(row1_reduced.tolist(), row2_reduced.tolist())
            result_df.loc[index1, index2] = sim      

    result_df = result_df.astype(float)
    return result_df

mat_FR5 = verification_last2_FR5()
if save_csv:
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/last2_FR5.csv'
    mat_FR5.to_csv(output_file, index=False, header=False)


def verification_last2_FR1():

    df = combined_binary()
    result_df = pd.DataFrame(index=df.index, columns=df.index)

    for index1,row1 in df.iterrows():
        for index2,row2 in df.iterrows():
            row1_reduced = row1[16:32] 
            row2_reduced = row2[16:32]  

            sim = jaccard_score(row1_reduced.tolist(), row2_reduced.tolist())
            result_df.loc[index1, index2] = sim      

    result_df = result_df.astype(float)
    return result_df


mat_FR1 = verification_last2_FR1()
if save_csv:
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/last2_FR1.csv'
    mat_FR1.to_csv(output_file, index=False, header=False)

sns.heatmap(mat, cmap='coolwarm', annot=False, fmt=".2f")
plt.show()