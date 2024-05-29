import pandas as pd
import os 
import re

name = 'L3'


csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
df = pd.read_csv(csv_file_path)


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
    
    files_list = files_list[:8]

    dfi = pd.DataFrame(files_list)
    print(dfi)

    df[key] = files_list

folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/Deconvolution L3/Extracted_Data'

get_file(folder_path, 'Deconvolution')

df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv', index=False)



print(df)