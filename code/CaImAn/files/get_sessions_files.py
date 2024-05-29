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

# def get_dat(dat_path_FR1, dat_path_FR5):
#     files_list = []

#     df1 = pd.read_csv(dat_path_FR1)
#     df5 = pd.read_csv(dat_path_FR5)

#     for index, row in df1.iterrows():
#         file_path = row.iloc[0]  
#         if 'ttl' in file_path:
#             files_list.append(file_path)



#     for index, row in df5.iterrows():
#         file_path = row.iloc[0]  
#         if 'ttl' in file_path:
#             files_list.append(file_path)

    
#     df['dat'] = files_list



def get_dat(dat_path_FR1, dat_path_FR5):
    files_list = []

    df1 = pd.read_csv(dat_path_FR1)
    df5 = pd.read_csv(dat_path_FR5)

    for index, row in df1.iterrows():
        file_path = row.iloc[0]  
        files_list.append(file_path)

    for index, row in df5.iterrows():
        file_path = row.iloc[0]  
        files_list.append(file_path)

    del files_list[0]

    df['dat'] = files_list
    


def neurons_files(folder_path,dat_path_FR1, dat_path_FR5, output_csv):

    get_dat(dat_path_FR1, dat_path_FR5)

    get_file(folder_path, 'Temporal_Components')
    get_file(folder_path, 'Time', 'Decay')
    get_file(folder_path, 'Peaks_TS')
    get_file(folder_path, 'Peaks_amplitude')
    get_file(folder_path, 'Spatial_Components')
    get_file(folder_path, 'Deconvolution')



    df.to_csv(output_csv, index=False)





#folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L2/Extracted_Data'
#folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/3 tis/Extracted_Data'
folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i2/Extracted_Data'

dat_path_FR1 = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR1_I2.csv'
dat_path_FR5 = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR5_I2.csv'
output_csv = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/I2.csv'

neurons_files(folder_path,dat_path_FR1, dat_path_FR5, output_csv)