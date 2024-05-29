import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from image_function import *
from behaviors_function import *


name = 'I2'

csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
df = pd.read_csv(csv_file_path)





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





def bin_label_7(label):
    if label == 'sequence':
        return 0
    if label == 'zone 1':
        return 1
    if label == 'zone 2':
        return 2
    if label == 'pellet':
        return 3
    if label == 'no pellet':
        return 4
    if label == 'explo 1':
        return 5
    if label == 'explo 2':
        return 6


def bin_label_4(label):
    if label == 'sequence':
        return 0
    if label == 'zone 1':
        return 1
    if label == 'zone 2':
        return 1
    if label == 'pellet':
        return 2
    if label == 'no pellet':
        return 2
    if label == 'explo 1':
        return 3
    if label == 'explo 2':
        return 3








def sum_traces(name):
    big_label_list = []

    long_reg_path = get_lr_path(name)
    long_reg_df = pd.read_csv(long_reg_path, header=None)

    row_numbers = extract_sessions(name)
    print(row_numbers)

    print(long_reg_df)

    if name == 'I6':
        long_reg_df = long_reg_df.drop(2, axis=1) 
        new_column_names = range(len(long_reg_df.columns))
        long_reg_df.columns = new_column_names  


    trace_dfs = {f'trace_df{id}': pd.read_csv(df.iloc[session,1], header=None) for id,session in enumerate(row_numbers)}
    print(trace_dfs)

    long_reg_df = long_reg_df.iloc[:,row_numbers]
    print(long_reg_df)

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

    sublists = [[] for _ in range(len(long_reg_df))]

    for id_session, session in enumerate(row_numbers):
        list_index = long_reg_df[session].tolist()
        trace_df = list(trace_dfs.values())[id_session]

        file_path = df.iloc[session, 0]
        all_actions_df = number_cycles(file_path)

        end_cut_column = int(all_actions_df.iloc[-1,0]/(1000*0.05))
        start_cut_column = int(all_actions_df.iloc[0,0]/(1000*0.05))

        for id_long_reg, elem in enumerate(list_index):

            if elem != 0:
                neuron_index = elem-1
                trace_list = trace_df.iloc[neuron_index,:].tolist()
            else:
                num_columns = trace_df.shape[1]
                trace_list = [0 for _ in range(num_columns)]


            trace_list = trace_list[start_cut_column + 1: end_cut_column + 1]
            #trace_list = trace_list[::20]
            sublists[id_long_reg].extend(trace_list)


        label_list = []
        for i in range(start_cut_column + 1, end_cut_column + 1):
            ms_time = i * 50
            time_limit_df = all_actions_df[all_actions_df['time'] < i * 50]

            label = time_limit_df.iloc[-1,1]
            label_list.append(bin_label_7(label))


        big_label_list.extend(label_list)

    new_df = pd.DataFrame(sublists)
    return big_label_list, new_df


sum_label_list, sum_df = sum_traces(name)

print(len(sum_label_list))
print(sum_df)






scaler = StandardScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(sum_df.T).T, columns=sum_df.columns)

trace_df = normalized_df.iloc[:, ::20]
sum_label_list = sum_label_list[::20]

sum_label_df = pd.DataFrame([sum_label_list])
sum_label_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/trace/labels/original_labels/{name}.csv', header=False, index=False)

correlation_matrix = trace_df.corr()
correlation_matrix.index = correlation_matrix.index * 50
correlation_matrix.columns = (correlation_matrix.columns * 50)

for i in range(2,11):

    num_clusters = i  
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(correlation_matrix)
    cluster_list = clusters.tolist()

    cluster_df = pd.DataFrame([cluster_list])
    cluster_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/trace/labels/new_labels/{name}/labels_{i}.csv', header=False, index=False)







color_list = ['blue', 'green', 'red', 'purple', 'orange', 'darkgray', 'gray']
cmap_custom = ListedColormap(color_list)

### PCA

pca = PCA(n_components=3)  
reduced_matrix = pca.fit_transform(trace_df.transpose())

# Step 3: Visualization
plt.figure(figsize=(8, 6))
plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=sum_label_list, cmap=cmap_custom)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()

plt.figure(figsize=(8, 6))
plt.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], c=cluster_list)
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()
plt.show()




