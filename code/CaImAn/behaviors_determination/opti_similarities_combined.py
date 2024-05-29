import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.cluster import hierarchy
import sys

name = 'I236'

direct = True
indirect = False

folder = 'trace'

save_csv = False

mat_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/all.csv'
combined_binary_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/combined_binary.csv'


mat = pd.read_csv(mat_path, header=None)

df = pd.read_csv(combined_binary_path, header=None)
len_df = len(df.iloc[0,:].tolist())

similarity_matrix = mat.values
linkage_matrix = hierarchy.linkage(similarity_matrix, method='average')



def plot_results():                                         # gives dendograms image

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    sns.heatmap(mat, ax=axs[0], cmap='coolwarm', annot=False, fmt=".2f")
    axs[0].set_title('Heatmap')

    hierarchy.dendrogram(linkage_matrix, ax=axs[1])
    axs[1].set_title('Dendrogram ')

    plt.tight_layout()
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/clusters_8/{name}/dendograms.png' , dpi=300)
    plt.show()
#plot_results()





def save_cluster_mouse(df, list, name1, name2, name3):
    
    print(df)
    new_list = [x for x in list if x != 13]

    str_list = [name1 if x == 10 else name2 if x == 11 else name3 if x == 12 else '' for x in new_list]
    print(len(new_list))

    df[2] = str_list
    print(df)

    if save_csv:
        output_file = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/neuron_class_mouse_6.csv'
        df.to_csv(output_file, index=False, header=False)
        

def enhanced_df_plot(linkage_matrix, nb_clusters, df):

    clusters = hierarchy.fcluster(linkage_matrix, nb_clusters, criterion='maxclust')

    list_all = [] 

    for i, label in enumerate(clusters):  
        row = [i,label]
        list_all.append(row)

    df_all = pd.DataFrame(list_all)

    sorted_df = df_all.sort_values(by=1, ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)


    list_insert = []
    previous_cat = None  

    for index, row in sorted_df.iterrows():
        if previous_cat is not None:  
            cat = row.iloc[1]
            if cat != previous_cat:
                list_insert.append(index)
        previous_cat = row.iloc[1]  


    list_neurons = sorted_df.iloc[:,0].tolist()

    for i,elem in enumerate(list_insert):
        list_neurons.insert(elem+i, -1)
    
    
    big_list = []
    for elem in list_neurons:
        if elem == - 1:
            row_of_minus_ones = pd.Series([9] * len_df)
            row_of_minus_ones.name = len(sorted_df) +1
            big_list.append(row_of_minus_ones) 
        else:
            big_list.append(df.iloc[elem,:])
            

    df_plot = pd.DataFrame(big_list)

    if direct:
        new_column = []
        for index,row in df_plot.iterrows():
            if index <= 99:
                new_column.append(10)
            if index >= 100 and index <= 206:
                new_column.append(11)
            if index >= 207 and index <= 307:
                new_column.append(12)
            if index >307:
                new_column.append(13)

        df_plot['mice'] = new_column

        save_cluster_mouse(sorted_df,new_column, 'I2', 'I3', 'I6')

    if indirect:
        new_column = []
        for index,row in df_plot.iterrows():
            if index <= 99:
                new_column.append(10)
            if index >= 100 and index <= 149:
                new_column.append(11)
            if index >= 150 and index <= 249:
                new_column.append(12)
            if index >249:
                new_column.append(13)

        df_plot['mice'] = new_column

        save_cluster_mouse(sorted_df,new_column, 'L0', 'L2', 'L3')

    

    for column_name, column_data in df_plot.items():
        column_data = column_data.apply(lambda x: x * (int(column_name) % 8 + 1) if x < 9 else x)
        df_plot[column_name] = column_data


    return df_plot


def comparative_nb(linkage_matrix, name1,name2,name3):

    #cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black'])
    cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black', 'orange', 'green','purple','black'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for i in range(6):  

        row = i // 3  
        col = i % 3   
        sns.heatmap(enhanced_df_plot(linkage_matrix, i+2, df), annot=False, cmap=cmap_custom, fmt=".2f", ax=axes[row, col])
        axes[row, col].set_title(f'All sessions ({i+2} clusters)')  # Set title for the subplot
        colorbar = axes[row, col].collections[0].colorbar
        # colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])  
        # colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$']) 
        colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13])  
        colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$', 'space', name1, name2, name3, 'space']) 

    plt.tight_layout()
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/clusters_8/{name}/nb_clusters_all_complete.png' , dpi=300)
    plt.show()

if name == 'I236':
    comparative_nb(linkage_matrix, 'I2','I3','I6')
if name == 'L023':
    comparative_nb(linkage_matrix, 'L0','L2','L3')


def plot_all(name1,name2,name3):
    cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black', 'orange', 'green','purple','black'])

    df_plot_all = enhanced_df_plot(linkage_matrix, 6, df)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df_plot_all, annot=False, cmap=cmap_custom, fmt=".2f", ax=ax)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13])  
    colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$', 'space', name1, name2, name3, 'space']) 
    plt.show()

# if direct:
#     plot_all('I2','I3','I6')
# if indirect:
#     plot_all('L0','L2','L3')

















