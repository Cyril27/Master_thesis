import pandas as pd

from image_function import *

csv_file_path = '/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat/dat_FR5_I6.csv'  
df = pd.read_csv(csv_file_path)

sequence = True
matrix = False


for index, row in df.iterrows():
    path = row.iloc[0]  
    
    if sequence:
        save_image(path)

    if matrix:
        save_matrix(path)
    






