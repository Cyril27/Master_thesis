import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, mutual_info_score, homogeneity_completeness_v_measure





name = 'L023'

if name == 'I236':
    cluster_decon_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/I236/neuron_class_mouse_6_average.csv'
if name == 'L023':
    cluster_decon_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/L023/neuron_class_mouse_6.csv'
cluster_decon_df = pd.read_csv(cluster_decon_path, header=None)

if name == 'I236':
    cluster_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/similarities_8/I236/neuron_class_mouse_6_average.csv'
if name == 'L023':
    cluster_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/similarities_8/L023/neuron_class_mouse_6.csv'
cluster_df = pd.read_csv(cluster_path, header=None)


cluster_decon_df = cluster_decon_df.sort_values(by=cluster_decon_df.columns[0])
cluster_decon_df = cluster_decon_df.reset_index(drop=True)

cluster_df = cluster_df.sort_values(by=cluster_df.columns[0])
cluster_df = cluster_df.reset_index(drop=True)

# print(cluster_decon_df)
# print(cluster_df)

def plot_comparison():

    matrix = np.zeros((cluster_decon_df.iloc[:,1].nunique(), cluster_df.iloc[:,1].nunique()))

    for (index1,row1), (index2,row2) in zip(cluster_df.iterrows(), cluster_decon_df.iterrows()):
        matrix[row2.iloc[1]-1, row1.iloc[1]-1] += 1

    row_sums = matrix.sum(axis=1)[:, np.newaxis]  
    matrix = matrix / row_sums

    plt.figure()
    sns.heatmap(matrix, cmap='Reds', annot=True, fmt=".2f", vmin=0, vmax=1)

    plt.xticks(np.arange(matrix.shape[1]) + 0.5, np.arange(matrix.shape[1]) + 1)
    plt.yticks(np.arange(matrix.shape[0]) + 0.5, np.arange(matrix.shape[0]) + 1)

    plt.xlabel('CaImAn cluster no.')
    plt.ylabel('Deconvolved cluster no.')

    plt.show()



plot_comparison()

def compute_vals():

    cluster_labels1 = cluster_decon_df.iloc[:, 1]
    cluster_labels2 = cluster_df.iloc[:, 1]

    print(f'Rand score : {rand_score(cluster_labels1,cluster_labels2)}')


    shuffled_df = cluster_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_cluster_labels2 = shuffled_df.iloc[:, 1]

    rand_score_shuffled = rand_score(cluster_labels1, shuffled_cluster_labels2)
    print(f'Rand score (shuffled): {rand_score_shuffled}')



    ari_score = adjusted_rand_score(cluster_labels1, cluster_labels2)

    ami_score = adjusted_mutual_info_score(cluster_labels1, cluster_labels2)

    fm_score = fowlkes_mallows_score(cluster_labels1, cluster_labels2)

    print(f'Adjusted Rand Index (ARI): {ari_score}')
    print(f'Adjusted Mutual Information (AMI): {ami_score}')
    print(f'Fowlkes-Mallows Index (FM): {fm_score}')


    print('###'*10)

    ari_score = adjusted_rand_score(cluster_labels1, shuffled_cluster_labels2)

    ami_score = adjusted_mutual_info_score(cluster_labels1, shuffled_cluster_labels2)

    fm_score = fowlkes_mallows_score(cluster_labels1, shuffled_cluster_labels2)

    print(f'Adjusted Rand Index (ARI): {ari_score}')
    print(f'Adjusted Mutual Information (AMI): {ami_score}')
    print(f'Fowlkes-Mallows Index (FM): {fm_score}')





cluster_labels1 = cluster_decon_df.iloc[:, 1].tolist()
cluster_labels2 = cluster_df.iloc[:, 1].tolist()

cluster_labels2_shuffled = np.copy(cluster_labels2)
np.random.shuffle(cluster_labels2_shuffled)


list = [rand_score(cluster_labels1,cluster_labels2),                        rand_score(cluster_labels1,cluster_labels2_shuffled),
        adjusted_rand_score(cluster_labels1, cluster_labels2),              adjusted_rand_score(cluster_labels1, cluster_labels2_shuffled),
        mutual_info_score(cluster_labels1, cluster_labels2),                mutual_info_score(cluster_labels1, cluster_labels2_shuffled),
        adjusted_mutual_info_score(cluster_labels1, cluster_labels2),       adjusted_mutual_info_score(cluster_labels1, cluster_labels2_shuffled),
        fowlkes_mallows_score(cluster_labels1, cluster_labels2),            fowlkes_mallows_score(cluster_labels1, cluster_labels2_shuffled)

]

pos = [0,1, 3,4, 6,7, 9,10, 12,13]
if name == 'I236':
    color_list = ['indianred', 'grey']
if name == 'L023':
    color_list = ['darkblue', 'grey']

label_list = ['Original', 'Shuffled']
for i in range(len(pos)):
    color = color_list[i%2]
    if i == 0 or i == 1:  # Add label for i = 0 and i = 1
        plt.bar(pos[i], list[i], color=color, label=label_list[i])
    else:
        plt.bar(pos[i], list[i], color=color)
    plt.xticks([0.5, 3.5, 6.5, 9.5, 12.5], ['Rand index', 'ARI', 'Mutual info.', 'AMI', 'Fowlkes-Mallows'])

plt.legend()
plt.tight_layout()
plt.show()