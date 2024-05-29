import os
import csv
import re

from behaviors_function import *

def extract_numeric_suffix(file_path):
    match = re.search(r'D(\d+)', file_path)
    if match:
        return int(match.group(1))
    return float('inf')  




def score_dat_files(folder_path, output_csv):
    dat_files = []
    
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if  file.endswith(".dat") :
                if 'I6' in file and not 'BASELINE' in file and not 'nc' in file and 'FR5' in file :
                    dat_files.append(os.path.join(root, file))

    
    sorted_file_paths = sorted(dat_files, key=extract_numeric_suffix)

    scored_paths = [(file, len(file.split(os.path.sep))) for file in sorted_file_paths]




    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['File']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file, score in scored_paths:

                    writer.writerow({'File': file})


folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/Imetronic .dat Sandra_FR_May2021'
output_csv = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR5_I6.csv'

score_dat_files(folder_path, output_csv)
