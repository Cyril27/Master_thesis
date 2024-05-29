import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

name = 'L3'

csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
df = pd.read_csv(csv_file_path)

filtered_df = df[df.iloc[:,0].str.contains('FR1')]
last_row_index = filtered_df.index[-1]

new_list = []
row_numbers = []

for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
    new_list.append(elem)
    row_numbers.append(index)

print(row_numbers)

for session in row_numbers:
    print(session)

    image_path = df.iloc[session, 5]
    image_df = pd.read_csv(image_path, header=None)
    print(image_df.shape)

    centroid_list = []

    for row in range(int(image_df.shape[1]/320)):

        image = image_df.iloc[:,320*(row):320*(row+1)]
        image_array = image.values
        non_zero_coords = np.argwhere(image_array > 0.05)

        centroid = np.mean(non_zero_coords, axis=0)
        centroid_list.append([centroid[1], 200-centroid[0]])


        # sns.heatmap(image, cmap='coolwarm', annot=False, fmt=".2f")
        # plt.show()


    centroid_df = pd.DataFrame(centroid_list)

    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/centroids/{name}/S{session+1}.csv'
    #centroid_df.to_csv(output_file, index=False, header=False)

    print(centroid_df)


    # plt.scatter(centroid_df[0], centroid_df[1])
    # plt.xlabel('X')
    # plt.ylabel('Y')

    # plt.xlim(0, 320)  
    # plt.ylim(0, 200)  
    # plt.show()