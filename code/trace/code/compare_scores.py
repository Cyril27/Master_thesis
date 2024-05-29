import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, mutual_info_score, homogeneity_completeness_v_measure




name = 'I236'


cluster_decon_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/I236/neuron_class_mouse_6_average.csv'
cluster_decon_df = pd.read_csv(cluster_decon_path, header=None)
cluster_trace_path = f'/Users/cyrilvanleer/Desktop/Thesis/trace/similarities_8/I236/neuron_class_mouse_6_average.csv'
cluster_trace_df = pd.read_csv(cluster_trace_path, header=None)
cluster_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/similarities_8/I236/neuron_class_mouse_6_average.csv'
cluster_df = pd.read_csv(cluster_path, header=None)


cluster_decon_df = cluster_decon_df.sort_values(by=cluster_decon_df.columns[0])
cluster_decon_df = cluster_decon_df.reset_index(drop=True)

cluster_trace_df = cluster_trace_df.sort_values(by=cluster_trace_df.columns[0])
cluster_trace_df = cluster_trace_df.reset_index(drop=True)

cluster_df = cluster_df.sort_values(by=cluster_df.columns[0])
cluster_df = cluster_df.reset_index(drop=True)


decon_labels = cluster_decon_df.iloc[:, 1].tolist()
trace_labels = cluster_trace_df.iloc[:, 1].tolist()
cluster_labels = cluster_df.iloc[:, 1].tolist()


list = [rand_score(cluster_labels,decon_labels),                        rand_score(cluster_labels,trace_labels),                      rand_score(decon_labels,trace_labels),
        adjusted_rand_score(cluster_labels, decon_labels),              adjusted_rand_score(cluster_labels, trace_labels),            adjusted_rand_score(decon_labels, trace_labels),
        mutual_info_score(cluster_labels, decon_labels),                mutual_info_score(cluster_labels, trace_labels),              mutual_info_score(decon_labels, trace_labels),
        adjusted_mutual_info_score(cluster_labels, decon_labels),       adjusted_mutual_info_score(cluster_labels, trace_labels),     adjusted_mutual_info_score(decon_labels, trace_labels),
        fowlkes_mallows_score(cluster_labels, decon_labels),            fowlkes_mallows_score(cluster_labels, trace_labels),          fowlkes_mallows_score(decon_labels, trace_labels)

]


pos = [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18]
if name == 'I236':
    color_list = ['indianred','lightcoral', 'rosybrown']
if name == 'L023':
    color_list = ['darkblue', 'grey']

label_list = ['CaImAn-MLspike', 'CaImAn-Traces', 'MLspike-Traces']
for i in range(len(pos)):
    color = color_list[i%3]
    if i == 0 or i == 1 or i==2:  # Add label for i = 0 and i = 1
        plt.bar(pos[i], list[i], color=color, label=label_list[i])
    else:
        plt.bar(pos[i], list[i], color=color)
    plt.xticks([1, 5, 9, 13, 17], ['Rand index', 'ARI', 'Mutual info.', 'AMI', 'Fowlkes-Mallows'])

plt.legend()
plt.tight_layout()
plt.show()