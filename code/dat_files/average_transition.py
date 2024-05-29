import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

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

fig, axs = plt.subplots(2, 4)
name_list = ['L0', 'L2', 'L3', 'I2', 'I3', 'I6']

for i in range(8):
    result_trans = np.zeros((7,7))


    for name in name_list:

        csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
        df = pd.read_csv(csv_file_path)

        row_numbers = extract_sessions(name)
        session = row_numbers[i]

        file_path = df.iloc[session,0]
        print(file_path)

        x = return_matrix(file_path)
        print(x)


        result_trans += x

    result_trans /= 6
    print(result_trans)


    names = ['Seq.', 'Zone 2', 'Pellet','No pellet', 'Explo 2','Explo 1','Zone 1']

    row = i // 4
    col = i %4

    print(row,col)

    sns.heatmap(result_trans, annot=True, cmap='Reds', square=True, xticklabels=names,  yticklabels=names, ax = axs[row,col], cbar=False, vmin=0, vmax=60
    )

    
    axs[row,col].set_xlabel('To behavior')
    if col == 0:
        axs[row,col].set_ylabel('From behavior')

    #plt.title('Transition matrix')
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.show()
        