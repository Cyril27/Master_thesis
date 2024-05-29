import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, AgglomerativeClustering, SpectralClustering, DBSCAN
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances

from sklearn.metrics import davies_bouldin_score




name='I236'
folder = 'deconvoled'

if name == 'L023':
    indirect = True
    direct = False
if name == 'I236':
    indirect = False
    direct = True

neurons_mat_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/all.csv'
mat = pd.read_csv(neurons_mat_path, header=None)


combined_binary_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{name}/combined_binary.csv'
df_binary = pd.read_csv(combined_binary_path, header=None)



#### Hierachichal average

similarity_matrix = mat.values
linkage_average_matrix = hierarchy.linkage(similarity_matrix, method='average')
hierarchical_average_clusters = hierarchy.fcluster(linkage_average_matrix, 6, criterion='maxclust')

print(hierarchical_average_clusters)

#### Hierachichal complete

similarity_matrix = mat.values
linkage_complete_matrix = hierarchy.linkage(similarity_matrix, method='complete')
hierarchical_complete_clusters = hierarchy.fcluster(linkage_complete_matrix, 6, criterion='maxclust')

print(hierarchical_complete_clusters)


#### Hierachichal ward

similarity_matrix = mat.values
linkage_ward_matrix = hierarchy.linkage(similarity_matrix, method='ward')
hierarchical_ward_clusters = hierarchy.fcluster(linkage_ward_matrix, 6, criterion='maxclust')

print(hierarchical_ward_clusters)



### Kmeans

kmeans = KMeans(n_clusters=5) 
kmeans.fit(mat)
kmeans_clusters = kmeans.labels_

print(kmeans_clusters)


### Agglomerative average

distance_matrix = 1 - mat.abs() 
agglomerative_average = AgglomerativeClustering(n_clusters=6, linkage='average').fit(distance_matrix)
agglomerative_average_labels = agglomerative_average.labels_

print(agglomerative_average_labels)


### Agglomerative complete

distance_matrix = 1 - mat.abs() 
agglomerative_complete = AgglomerativeClustering(n_clusters=6, linkage='complete').fit(distance_matrix)
agglomerative_complete_labels = agglomerative_complete.labels_

print(agglomerative_complete_labels)


### Agglomerative ward

distance_matrix = 1 - mat.abs() 
agglomerative_ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(distance_matrix)
agglomerative_ward_labels = agglomerative_ward.labels_

print(agglomerative_ward_labels)



### Spectral

similarity_matrix = 1 - mat.abs() 
spectral_clustering = SpectralClustering(n_clusters=6, affinity='nearest_neighbors', random_state=42)
spectral_labels = spectral_clustering.fit_predict(similarity_matrix)

print(spectral_labels)


### Affinity propagation

preference = -40.0  
affinity_propagation = AffinityPropagation(damping=0.5, preference=preference)
affinity_labels = affinity_propagation.fit_predict(mat)

print(affinity_labels)


### Mean shift


bandwidth = 1.8  # Adjust this value as needed
mean_shift = MeanShift(bandwidth=bandwidth)
meanshift_labels = mean_shift.fit_predict(mat)

print(meanshift_labels)



def enhanced_df_plot(clusters, df):

    clusters = clusters.tolist()

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
            row_of_minus_ones = pd.Series([9] * 64)
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


    for column_name, column_data in df_plot.items():
        column_data = column_data.apply(lambda x: x * (int(column_name) % 8 + 1) if x < 9 else x)
        df_plot[column_name] = column_data


    return df_plot

def comparative_nb(cluster, i, title):

    #cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black'])
    cmap_custom = sns.color_palette(['white', 'brown','indianred', 'lightcoral','darksalmon','blue', 'royalblue','cornflowerblue','skyblue',  'black', 'orange', 'green','purple','black'])
    
    row = i // 3  
    col = i % 3   
    print(row,col)
    sns.heatmap(enhanced_df_plot(cluster, df_binary), annot=False, cmap=cmap_custom, fmt=".2f", ax=axes[row, col], cbar=False)
    axes[row, col].set_title(f'{title} method')  # Set title for the subplot

    #colorbar = axes[row, col].collections[0].colorbar
    # colorbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13])  
    # colorbar.set_ticklabels([r'sequence $\uparrow$', r'zones $\uparrow$', r'food $\uparrow$', r'explo $\uparrow$', r'sequence $\downarrow$', r'zones $\downarrow$', r'food $\downarrow$', r'explo $\downarrow$', 'space', 'I2', 'I3', 'I6', 'space']) 

    #plt.tight_layout()
    #plt.show()







fig, axes = plt.subplots(3, 3, figsize=(9, 7))

def plot_all_methods():
    clusters_list = [ hierarchical_average_clusters, hierarchical_complete_clusters, hierarchical_ward_clusters , kmeans_clusters, agglomerative_average_labels, agglomerative_complete_labels, agglomerative_ward_labels, spectral_labels, affinity_labels]
    method_list = ['Hierarchical (avg)','Hierarchical (comp)','Hierarchical (ward)', 'Kmeans', 'Agglo. (avg)','Agglo. (comp)', 'Agglo. (ward)', 'Spectral', 'Affinity']


    for i,cluster in enumerate(clusters_list):
        comparative_nb(cluster,i, method_list[i])


    plt.tight_layout()
    plt.show()


plot_all_methods()






def compute_rand_index(clusters_list):
    num_clusterings = len(clusters_list)
    rand_indices = np.zeros((num_clusterings, num_clusterings))
    for i in range(num_clusterings):
        for j in range(num_clusterings):
            rand_indices[i, j] = adjusted_rand_score(clusters_list[i], clusters_list[j])
    return rand_indices

def compare_kmeans():
    clusters_list = []
    for i in range(9):

        kmeans = KMeans(n_clusters=5) 
        kmeans.fit(mat)
        kmeans_clusters = kmeans.labels_

        comparative_nb(kmeans_clusters,i, f'KMeans {i+1}')

        clusters_list.append(kmeans_clusters.tolist())


    plt.tight_layout()
    #plt.show()

    rand_indices = compute_rand_index(clusters_list)

    plt.figure()
    ax = sns.heatmap(rand_indices, annot=True, cmap='Blues', fmt=".2f")
    plt.title("Rand Index between Kmeans Clusterings")
    plt.xlabel("Kmeans clustering No.")
    plt.ylabel("Kmeans clustering No.")

    ax.set_xticks(np.arange(0.5, len(clusters_list) + 0.5, 1))
    ax.set_xticklabels(np.arange(1, len(clusters_list) + 1, 1))
    ax.set_yticks(np.arange(0.5, len(clusters_list) + 0.5, 1))
    ax.set_yticklabels(np.arange(1, len(clusters_list) + 1, 1))

    plt.show()






#compare_kmeans()











def davies_bouldin_index(cluster_labels, correlation_matrix):
    # Compute pairwise distances between clusters based on the correlation matrix
    n_clusters = len(np.unique(cluster_labels))
    cluster_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            mask_i = cluster_labels == i
            mask_j = cluster_labels == j
            mean_corr_i = correlation_matrix[mask_i].mean(axis=0)
            mean_corr_j = correlation_matrix[mask_j].mean(axis=0)
            cluster_distances[i, j] = np.linalg.norm(mean_corr_i - mean_corr_j)
            cluster_distances[j, i] = cluster_distances[i, j]

    # Compute the Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(correlation_matrix, cluster_labels)
    
    return davies_bouldin, cluster_distances

# davies_bouldin, cluster_distances = davies_bouldin_index(hierarchical_clusters, mat)
# print("Davies-Bouldin Index:", davies_bouldin)

# cluster_labels_shuffled = np.array(hierarchical_clusters)  # Convert to numpy array if not already
# np.random.shuffle(cluster_labels_shuffled)

# davies_bouldin, cluster_distances = davies_bouldin_index(cluster_labels_shuffled, mat)
# print("Davies-Bouldin Index:", davies_bouldin)












