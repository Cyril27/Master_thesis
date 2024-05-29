import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


big_name = 'all'


if big_name == 'I236':
    name_list = ['I2', 'I3', 'I6']
if big_name == 'L023':
    name_list = ['L0', 'L2', 'L3']
if big_name == 'all':
    name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']




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



def quality_percentage():
    result = []
    for name in name_list:
        lr_path = get_lr_path(name)
        long_reg_df = pd.read_csv(lr_path, header=None)


        # if name == 'I6':
        #     long_reg_df = long_reg_df.drop(2, axis=1)  
        #     long_reg_df.columns = list(range(len(long_reg_df.columns)))
        

        subset_20 = long_reg_df[long_reg_df.apply(lambda row: 0.0 <= (row != 0).sum() / len(row) < 0.2, axis=1)]
        subset_40 = long_reg_df[long_reg_df.apply(lambda row: 0.2 <= (row != 0).sum() / len(row) < 0.4, axis=1)]
        subset_60 = long_reg_df[long_reg_df.apply(lambda row: 0.4 <= (row != 0).sum() / len(row) < 0.6, axis=1)]
        subset_80 = long_reg_df[long_reg_df.apply(lambda row: 0.6 <= (row != 0).sum() / len(row) < 0.8, axis=1)]
        subset_100 = long_reg_df[long_reg_df.apply(lambda row: 0.8 <= (row != 0).sum() / len(row) <= 1, axis=1)]

        row = [len(subset_20)/len(long_reg_df),  len(subset_40)/len(long_reg_df),  len(subset_60)/len(long_reg_df), len(subset_80)/len(long_reg_df), len(subset_100)/len(long_reg_df)]
        result.append(row)

    result_df = pd.DataFrame(result)

    print(result_df)


    mean_values = result_df.mean()

    
    plt.bar(mean_values.index, mean_values, label='Mean', color = 'darkblue', alpha=0.7)
    for column in result_df.columns:
        plt.scatter([column] * len(result_df), result_df[column], color='white', edgecolors='black', alpha=1)

    plt.xlabel('Percentage of sessions in which the neuron is found')
    plt.ylabel('Percentage of registration sequences')
    #plt.title('Mean of Each Column with Corresponding Points')
    plt.xticks(range(len(result_df.columns)), ['0 to 20', '20 to 40', '40 to 60', '60 to 80', '80 to 100'])

    plt.show()




result = []
for name in name_list:
    lr_path = get_lr_path(name)
    long_reg_df = pd.read_csv(lr_path, header=None)
    row_numbers = extract_sessions(name)

    if name == 'I6':
        long_reg_df = long_reg_df.drop(2, axis=1)  
        long_reg_df.columns = list(range(len(long_reg_df.columns)))

    
    long_reg_df = long_reg_df.iloc[:,row_numbers]
    long_reg_df.columns = range(long_reg_df.shape[1])
    

    non_zero_counts = (long_reg_df != 0).sum(axis=1)


    rows_01 = long_reg_df[(non_zero_counts >= 0) & (non_zero_counts <= 1)]
    rows_23 = long_reg_df[(non_zero_counts >= 2) & (non_zero_counts <= 3)]
    rows_4 = long_reg_df[(non_zero_counts == 4)]
    rows_56 = long_reg_df[(non_zero_counts >= 5) & (non_zero_counts <= 6)]
    rows_78 = long_reg_df[(non_zero_counts >= 7) & (non_zero_counts <= 8)]
    

    result_list = [len(rows_01)/len(long_reg_df), len(rows_23)/len(long_reg_df), len(rows_4)/len(long_reg_df), len(rows_56)/len(long_reg_df), len(rows_78)/len(long_reg_df)]
    result.append(result_list)

result_df = pd.DataFrame(result)

print(result_df)


mean_values = result_df.mean()

markers = ['o', '^', 'v', 's', 'p', '*', 'h', 'D']

# Plot the bar plot
plt.bar(mean_values.index, mean_values, label='Mean', color = 'darkblue', alpha=0.7)

# Plot the corresponding points for each column
# for column in result_df.columns:
#     plt.scatter([column] * len(result_df), result_df[column], color='white', edgecolors='black', alpha=1)

for i, column in enumerate(result_df.columns):
    column = result_df[column].tolist()
    for j,elem in enumerate(column):
        plt.scatter(i, elem, color='none', edgecolors='black', alpha=1,)


plt.xlabel('Number of sessions in which the neuron is found (out of 8 sessions)')
plt.ylabel('Percentage of registration sequences')
#plt.title('Mean of Each Column with Corresponding Points')
plt.xticks(range(len(result_df.columns)), ['0 to 1', '2 to 3', '4', '5 to 6', '7 to 8'])

plt.show()



    