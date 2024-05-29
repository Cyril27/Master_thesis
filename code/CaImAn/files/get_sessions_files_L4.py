import os
import csv
import pandas as pd

df = pd.DataFrame() 


def get_dat(dat_path_FR1,dat_path_FR5):
    df1 = pd.read_csv(dat_path_FR1)
    df5 = pd.read_csv(dat_path_FR5)

    files_list = []
    files_list.append(df1.iloc[6,0])

    for i in range(8,18):
        files_list.append(df1.iloc[i,0])

    for i in range(2,10):
        files_list.append(df5.iloc[i,0])

    df['dat'] = files_list



def get_file(folder_path1, folder_path2, key, key2='fhrfhri'):
    files_list = []
    for root, dirs, files in os.walk(folder_path1):
        for file in files:
            if  file.endswith(".csv") :
                if  key in file and not key2 in file:
                    files_list.append(os.path.join(root, file))

    

    for root, dirs, files in os.walk(folder_path2):
        for file in files:
            if  file.endswith(".csv") :
                if  key in file and not key2 in file and not 'Sess1_' in file:
                    files_list.append(os.path.join(root, file))

    df[key] = files_list

   
   





def neurons_files(folder_path1, folder_path2,dat_path_FR1, dat_path_FR5, output_csv):

    get_dat(dat_path_FR1, dat_path_FR5)

    get_file(folder_path1, folder_path2, 'Temporal_Components')
    get_file(folder_path1, folder_path2, 'Time', 'Decay')
    get_file(folder_path1, folder_path2, 'Peaks_TS')
    get_file(folder_path1, folder_path2, 'Peaks_amplitude')
    get_file(folder_path1, folder_path2, 'Spatial_Components')


    df.to_csv(output_csv, index=False)

dat_path_FR1 = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR1_L4.csv'
dat_path_FR5 = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR5_L4.csv'
folder_path1 = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR1 CUED/Longit FR1 L4/Extracted_Data'
folder_path2 = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L4/Extracted_Data'


output_csv = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/L4.csv'

neurons_files(folder_path1, folder_path2,dat_path_FR1, dat_path_FR5, output_csv)