import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.cluster import hierarchy
import sys

name = 'L3'

direct = False
indirect = False

folder = 'deconvoled'




mat_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/all.csv'
mat_FR5_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/last2_FR5.csv'
mat_FR1_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/last2_FR1.csv'

combined_binary_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/combined_binary.csv'

save_csv = False

mat = pd.read_csv(mat_path, header=None)
mat_FR5 = pd.read_csv(mat_FR5_path, header=None)
mat_FR1 = pd.read_csv(mat_FR1_path, header=None)

df = pd.read_csv(combined_binary_path, header=None)
len_df = len(df.iloc[0,:].tolist())

similarity_matrix = mat.values
linkage_matrix = hierarchy.linkage(similarity_matrix, method='complete')

similarity_matrix_FR5 = mat_FR5.values
linkage_matrix_FR5 = hierarchy.linkage(similarity_matrix_FR5, method='complete')

similarity_matrix_FR1 = mat_FR1.values
linkage_matrix_FR1 = hierarchy.linkage(similarity_matrix_FR1, method='complete')

def plot_results():                                         # gives dendograms image

    fig, axs = plt.subplots(2, 3, figsize=(10, 7))

    sns.heatmap(mat, ax=axs[0, 0], cmap='coolwarm', annot=False, fmt=".2f")
    axs[0, 0].set_title('Heatmap')

    sns.heatmap(mat_FR5, ax=axs[0, 1], cmap='coolwarm', annot=False, fmt=".2f")
    axs[0, 1].set_title('Heatmap FR5')

    sns.heatmap(mat_FR1, ax=axs[0, 2], cmap='coolwarm', annot=False, fmt=".2f")
    axs[0, 2].set_title('Heatmap FR1')

    hierarchy.dendrogram(linkage_matrix, ax=axs[1, 0])
    axs[1, 0].set_title('Dendrogram ')

    hierarchy.dendrogram(linkage_matrix_FR5, ax=axs[1, 1])
    axs[1, 1].set_title('Dendrogram FR5')

    hierarchy.dendrogram(linkage_matrix_FR1, ax=axs[1, 2])
    axs[1, 2].set_title('Dendrogram FR1')

    plt.tight_layout()
    plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/clusters_8/{name}/dendograms.png' , dpi=300)
    plt.show()

plot_results()


def comparison(threshold1, threshold2, num_classes):        #bit useless now

    distance_threshold1 = threshold1
    clusters1 = hierarchy.fcluster(linkage_matrix, distance_threshold1, criterion='distance')
    #clusters1 = hierarchy.fcluster(linkage_matrix, 6, criterion='maxclust')

    distance_threshold2 = threshold2
    clusters2 = hierarchy.fcluster(linkage_matrix_FR5, distance_threshold2, criterion='distance')

    matrix = np.zeros((num_classes,num_classes))

    for i, (label1,label2) in enumerate(zip(clusters1,clusters2)):
        #print(i,label1,label2)
        matrix[label1-1 ,label2-1] += 1

    return matrix

def multiple_comparisons():                                 #bit useless now

    m2 = comparison(2.5, 3.5, 2)
    m3 = comparison(2.3, 3.1, 3)
    m4 = comparison(2.1, 3.0, 4)
    m5 = comparison(2.045, 2.975, 5)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    sns.heatmap(m2, cmap='coolwarm', ax=axs[0, 0], annot=True, fmt=".2f")
    sns.heatmap(m3, cmap='coolwarm', ax=axs[0, 1], annot=True, fmt=".2f")
    sns.heatmap(m4, cmap='coolwarm', ax=axs[1, 0], annot=True, fmt=".2f")
    sns.heatmap(m5, cmap='coolwarm', ax=axs[1, 1], annot=True, fmt=".2f")

    plt.tight_layout()
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/neurons/images/correlation/L0/multiple.png' , dpi=300)
    plt.show()


# the 3 used to compared the results of clustering and redo a clustering from the most common pairs

def cut(num_cluster1, num_cluster2):

    clusters1 = hierarchy.fcluster(linkage_matrix, num_cluster1, criterion='maxclust')
    clusters2 = hierarchy.fcluster(linkage_matrix_FR5, num_cluster2, criterion='maxclust')

    matrix = np.zeros((num_cluster1,num_cluster2))

    list = [] 

    for i, (label1,label2) in enumerate(zip(clusters1,clusters2)):
        row = [i,label1,label2]
        list.append(row)
        #print(i,label1,label2)
        #print(matrix)
        matrix[label1-1 ,label2-1] += 1
        
    df_cluster_pair = pd.DataFrame(list)
    return matrix, df_cluster_pair
 
def find_best_pairs(mat):

    total_sum = np.sum(mat)
    mat = mat/total_sum

    values = []
    x_positions = []
    y_positions = []

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            values.append(mat[i, j])
            x_positions.append(i)
            y_positions.append(j)

    df = pd.DataFrame({'Value': values, 'X Position': x_positions, 'Y Position': y_positions})
    sorted_df = df.sort_values(by='Value', ascending=False)
    sorted_df = sorted_df.reset_index(drop=True)

    cumulative_sum = sorted_df['Value'].cumsum()
    cut_index = cumulative_sum[cumulative_sum > 0.8].index[0]
    cut_df = sorted_df.iloc[:cut_index+1]

    #return cut_df
    return sorted_df

def plot_clustered_sequences():
    mat, df_cluster_pair = cut(3,3)
    #print(mat)
    best_df = find_best_pairs(mat)
    #print(best_df)
    #print(df_cluster_pair)

    big_list = []
    for index,row in best_df.iterrows():
        index1 = row.iloc[1] +1
        index2 = row.iloc[2] +1

        print(index1,index2)
        filtered_values = df_cluster_pair.loc[(df_cluster_pair[1] == index1) & (df_cluster_pair[2] == index2), 0]
        list = filtered_values.tolist()


        list_df = []
        for elem in list:
            list_df.append(df.iloc[elem,:])
            big_list.append(df.iloc[elem,:])

        df_compare = pd.DataFrame(list_df)
        print(df_compare)
        

    df_plot = pd.DataFrame(big_list)

    sns.heatmap(df_plot, annot=False, fmt=".2f")
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/neurons/images/correlation/L0/3_4.png' , dpi=300)
    plt.show()
#plot_clustered_sequences()


def save_cluster_mouse(df, list, name1, name2, name3):
    
    print(df)
    new_list = [x for x in list if x != 13]

    str_list = [name1 if x == 10 else name2 if x == 11 else name3 if x == 12 else '' for x in new_list]
    print(len(new_list))

    df[2] = str_list
    print(df)

    if save_csv:
        output_file = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/similarities_8/L023/neuron_class_mouse_6_average.csv'
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

        df_plot['mouses'] = new_column

        #save_cluster_mouse(sorted_df,new_column, 'I2', 'I3', 'I6')

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

        df_plot['mouses'] = new_column

        #save_cluster_mouse(sorted_df,new_column, 'L0', 'L2', 'L3')

    

    for column_name, column_data in df_plot.items():
        column_data = column_data.apply(lambda x: x * (int(column_name) % 8 + 1) if x < 9 else x)
        df_plot[column_name] = column_data


    return df_plot


def comparative_plot():

    df_plot_all = enhanced_df_plot(linkage_matrix, 5, df)
    df_plot_FR1 = enhanced_df_plot(linkage_matrix_FR1, 4, df)
    df_plot_FR5 = enhanced_df_plot(linkage_matrix_FR5, 4, df)


    cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black'])
    name_list = ['All sessions', 'last 2 FR1','last 2 FR5']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, df_i in enumerate([df_plot_all, df_plot_FR1, df_plot_FR5]):  
        sns.heatmap(df_i, annot=False, cmap=cmap_custom, fmt=".2f", ax=axes[i])
        axes[i].set_title(name_list[i])
        colorbar = axes[i].collections[0].colorbar
        colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])  
        colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$']) 
        

    plt.tight_layout()
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/clusters_8/{name}/compare.png' , dpi=300)
    plt.show()

#comparative_plot()


def comparative_nb(linkage_matrix, output_name):

    cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black'])
    #cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black', 'orange', 'green','purple','black'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    for i in range(6):  

        row = i // 3  
        col = i % 3   
        sns.heatmap(enhanced_df_plot(linkage_matrix, i+2, df), annot=False, cmap=cmap_custom, fmt=".2f", ax=axes[row, col])
        axes[row, col].set_title(f'All sessions ({i+2} clusters)')  # Set title for the subplot
        colorbar = axes[row, col].collections[0].colorbar
        colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8])  
        colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$']) 
        # colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13])  
        # colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$', 'space', 'I2', 'I3', 'I6', 'space']) 

    plt.tight_layout()
    plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/clusters_8/{name}/{output_name}.png' , dpi=300)
    plt.show()

comparative_nb(linkage_matrix, 'nb_clusters_all')
comparative_nb(linkage_matrix_FR1, 'nb_clusters_FR1')
comparative_nb(linkage_matrix_FR5, 'nb_clusters_FR5')


def plot_all(name1,name2,name3):
    cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black', 'orange', 'green','purple','black'])

    df_plot_all = enhanced_df_plot(linkage_matrix, 6, df)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_plot_all, annot=False, cmap=cmap_custom, fmt=".2f", ax=ax)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13])  
    colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$', 'space', name1, name2, name3, 'space']) 
    plt.show()

# if direct:
#     plot_all('I2','I3','I6')
# if indirect:
#     plot_all('L0','L2','L3')

















