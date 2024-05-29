import os
import csv


def score_dat_files(folder_path, output_csv):
    dat_files = []
    
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if  file.endswith(".dat") :
                if not 'FM' in file and not 'NOCUE' in file and 'TTL' in file:
                    dat_files.append(os.path.join(root, file))

    
    scored_paths = [(file, len(file.split(os.path.sep))) for file in dat_files]

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['File']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file, score in scored_paths:

            writer.writerow({'File': file})
            


folder_path = '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/Imetronic .dat Jul-Ag2020'
output_csv = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_TTL/TTL_all.csv'

score_dat_files(folder_path, output_csv)
