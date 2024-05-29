import os
import csv
import pandas as pd
import re

df = pd.DataFrame() 


def extract_numeric_suffix(file_path):
    match = re.search(r'Sess(\d+)', file_path)
    if match:
        return int(match.group(1))
    return float('inf') 

def get_file(folder_path, key, key2='fhrfhri'):
    files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if  file.endswith(".csv") :
                if  key in file and not key2 in file:
                    files_list.append(os.path.join(root, file))

    files_list.sort(key=extract_numeric_suffix)

    df[key] = files_list

def get_dat(dat_path_FR1, dat_path_FR5):
    files_list = []

    df1 = pd.read_csv(dat_path_FR1)
    df5 = pd.read_csv(dat_path_FR5)

    list_fr1 = [10,11,17,18,20,21,22,23]
    for row in list_fr1:
        files_list.append(df1.iloc[row,0])

    list_fr5 = [0,1,5,6,7]
    for row in list_fr5:
        files_list.append(df5.iloc[row,0])

    sorted_file_paths = sorted(files_list, key=extract_numeric_suffix)
    df['dat'] = sorted_file_paths

    

    


def neurons_files(folder_path,dat_path_FR1, dat_path_FR5, output_csv):

    get_dat(dat_path_FR1, dat_path_FR5)

    get_file(folder_path, 'Temporal_Components')
    get_file(folder_path, 'Time', 'Decay')
    get_file(folder_path, 'Peaks_TS')
    get_file(folder_path, 'Peaks_amplitude')
    get_file(folder_path, 'Spatial_Components')


    df.to_csv(output_csv, index=False)





folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_a2ap4/Extracted_Data'
dat_path_FR1 = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR1_P4.csv'
dat_path_FR5 = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR5_P4.csv'
output_csv = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/P4.csv'

neurons_files(folder_path,dat_path_FR1, dat_path_FR5, output_csv)


