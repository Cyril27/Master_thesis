import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from image_function import *



big_name = 'L023'
name = 'L2'

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

long_reg_path = get_lr_path(name)
long_reg_df = pd.read_csv(long_reg_path, header = None)

if name == 'I6':
    long_reg_df = long_reg_df.drop(2, axis=1) 
    new_column_names = range(len(long_reg_df.columns))
    long_reg_df.columns = new_column_names  



def get_all_index(cluster, session):

    #### Original neuron index


    if big_name == 'I236':
        data_path = '/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/I236/neuron_class_mouse_6_average.csv'

    if big_name == 'L023':
        data_path = '/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/L023/neuron_class_mouse_6.csv'
    
    data_df = pd.read_csv(data_path, header = None)

    long_reg_index = data_df[(data_df.iloc[:,2] == name) & (data_df.iloc[:,1] == cluster)].iloc[:,0].tolist()           # Filter over the session and the name
    neuron_index_1 = long_reg_df.iloc[long_reg_index,:].iloc[:,session].tolist()                                        # Filter over the cluster
    neuron_index_1 = [value-1 for value in neuron_index_1 if value != 0]
    #print(neuron_index_1)

    ### New candidates neuron index

    new_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/more_neurons/{name}_all/S{session+1}.csv'                    # Filter over the session and the name
    new_df = pd.read_csv(new_path, header = None)                                                                 

    neuron_index_2 = new_df[new_df.iloc[:,2] == cluster].iloc[:,1].tolist()                                             # Filter over the cluster
    #print(neuron_index_2)

    all_index = sorted(neuron_index_1 +neuron_index_2)

    return all_index




def perform_svm(decon_index, session):

    decon_path = df.iloc[session, 6]
    decon_df = pd.read_csv(decon_path, header=None)
    decon_df = decon_df.iloc[decon_index,:].reset_index(drop=True)

    

    ### Get the label for each time frame

    file_path = df.iloc[session, 0]
    all_actions_df = number_cycles(file_path)

    end_cut_column = int(all_actions_df.iloc[-1,0]/(1000*0.05))
    decon_df = decon_df.iloc[:,:end_cut_column+1]

    start_cut_column = int(all_actions_df.iloc[0,0]/(1000*0.05))
    decon_df = decon_df.iloc[:,start_cut_column+1:]


    label_list = []
    for i in range(start_cut_column + 1, end_cut_column + 1):
        ms_time = i * 50
        time_limit_df = all_actions_df[all_actions_df['time'] < i * 50]

        label = time_limit_df.iloc[-1,1]
        label_list.append(label)

    label_encoder = LabelEncoder()
    numerical_label_list = label_encoder.fit_transform(label_list)

    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label Mapping:")
    for label, encoded_label in label_mapping.items():
        print(f"{label}: {encoded_label}")


    print(decon_df)



    ### Perfom svm

    X = decon_df.T.values
    y = numerical_label_list



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    
    svm_classifier = SVC(kernel='linear', random_state=42, decision_function_shape='ovo')
    #svm_classifier = LinearSVC(random_state=42, max_iter=10000, dual=True, multi_class='ovr')
    svm_classifier.fit(X_train, y_train)


    accuracy = svm_classifier.score(X_test, y_test)
    print("Accuracy:", accuracy)

    y_pred = svm_classifier.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)
    print(cm)


    


    

    # label_index = 4  

    # if len(cm) == 7:
    #     total = cm[label_index].sum()
    #     correct_pred = cm[label_index, label_index]
    #     perc = (correct_pred / total) * 100
    #     print(f'Correct pred of sequence: {perc}')
    
    # else:
    #     print('less behaviors')
    #     perc = 0.0

    label_list = ['explo 1', 'explo 2', 'no pellet', 'pellet', 'sequence', 'zone 1', 'zone 2']

    if len(cm) != len(label_list):
        return [float('nan')] * 6

    total = [0]*7
    correctly_pred = [0]*7
    for i,elem in enumerate(label_list):
        if elem in label_mapping:
            index = label_mapping[elem]

            total_i = cm[index].sum()
            total[i] = total_i

            correctly_pred_i = cm[index,index]
            correctly_pred[i]= correctly_pred_i


    # Prediction sequence
    if total[4] != 0:
        pred_seq = correctly_pred[4]/total[4]
    else:
        pred_seq = 0
    # Prediction zones
    if total[5]+total[6] != 0:
        pred_zones = (correctly_pred[5] + correctly_pred[6])/(total[5] + total[6])
    else:
        pred_zones = 0
    # Prediction food
    pred_food = (correctly_pred[2] + correctly_pred[3])/(total[2] + total[3])
    # Prediction explo
    pred_explo = (correctly_pred[0] + correctly_pred[1])/(total[0] + total[1])
    # Prediction global
    pred_global = (np.sum(correctly_pred)) / (np.sum(total))
    # total len
    total_len = np.sum(total)

    output_vector = [pred_seq, pred_zones, pred_food, pred_explo, pred_global, total_len]

    print(output_vector)


    return output_vector





row_numbers = extract_sessions(name)

num_mat = np.zeros((6,8))
seq_mat = np.zeros((6,8))
zone_mat = np.zeros((6,8))
food_mat = np.zeros((6,8))
explo_mat = np.zeros((6,8))
global_mat = np.zeros((6,8))


for cluster in range(1,7):
    for j,session in enumerate(row_numbers):

        

        print(f'cluster {cluster}, session {j+1}')
        decon_index = get_all_index(cluster, session)
        output_vector = perform_svm(decon_index,session)

        
        seq_mat[cluster-1, j] = output_vector[0]
        zone_mat[cluster-1, j] = output_vector[1]
        food_mat[cluster-1, j] = output_vector[2]
        explo_mat[cluster-1, j] = output_vector[3]
        global_mat[cluster-1, j] = output_vector[4]
        num_mat[cluster-1, j] = output_vector[5]

        print(seq_mat)
        

zone_df = pd.DataFrame(zone_mat)
zone_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_zone/{big_name}/decon_6_all/{name}.csv' , index=False, header=False)

food_df = pd.DataFrame(food_mat)
food_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_food/{big_name}/decon_6_all/{name}.csv' , index=False, header=False)

explo_df = pd.DataFrame(explo_mat)
explo_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_explo/{big_name}/decon_6_all/{name}.csv' , index=False, header=False)


seq_df = pd.DataFrame(seq_mat)
seq_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_seq/{big_name}/decon_6_all/{name}.csv' , index=False, header=False)

global_df = pd.DataFrame(global_mat)
global_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_global/{big_name}/decon_6_all/{name}.csv' , index=False, header=False)






plt.figure()
heatmap = sns.heatmap(global_mat, annot=True, fmt=".2f", cmap='Blues', cbar=False)

plt.figure()
heatmap = sns.heatmap(seq_mat, annot=True, fmt=".2f", cmap='Blues', cbar=False)

plt.figure()
heatmap = sns.heatmap(food_mat, annot=True, fmt=".2f", cmap='Blues', cbar=False)

plt.show()