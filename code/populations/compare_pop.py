import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# color_list = ['cornflowerblue', 'indianred', 'mediumseagreen', 'mediumpurple', 'palegoldenrod', 'darkgray']

# for i in range(6):
#     plt.bar(1,1, label=f'Cluster {i+1}', color = color_list[i])
# plt.legend()
# plt.tight_layout()
# plt.show()





big_name = 'L023'


def plot_percentage(num_clusters, behav):

    if big_name == 'I236':
        name_list = ['I2', 'I3', 'I6']
    if big_name == 'L023':
        name_list = ['L0', 'L2', 'L3']

    big_list = []
    for name in name_list:
        list = []
        path = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{behav}/{big_name}/decon_{num_clusters}_all/{name}.csv'
        df = pd.read_csv(path, header =None)

        # if name == 'I2':
        #     df = df/100


        for column_name in df.columns:
            column_data = df[column_name].tolist()
            list.extend(column_data)

        big_list.append(list)

    result_df = pd.DataFrame(big_list)
    print(result_df)


    


    pos = []
    for i in range(8):
        for j in range(num_clusters):
        #sub = [0+i*(num_clusters+1), 1+i*(num_clusters+1)]
            pos.append(j+i*(num_clusters+1))

    print(pos)

    if num_clusters == 2:
        color_list = ['cornflowerblue', 'indianred']
    if num_clusters == 3:
        color_list = ['cornflowerblue', 'indianred', 'mediumseagreen']
    if num_clusters == 6:
        color_list = ['cornflowerblue', 'indianred', 'mediumseagreen', 'mediumpurple', 'palegoldenrod', 'darkgray']


    name_color = ['blue', 'red', 'green']

    marker_list = ['o', 'v', 'D']


    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(result_df.shape[1]):
        col = result_df.iloc[:, i].dropna().tolist()
        ax.bar(pos[i], np.mean(col), label=f'Session {i+1}', color = color_list[i%num_clusters], edgecolor = 'none')
        for j,val in enumerate(col):
            ax.scatter([pos[i]], val, color='none', edgecolor='black', s=8, marker=marker_list[j])

    ax.set_xticks(pos)
    ax.set_xticklabels([f'Cluster {p%num_clusters +1}' for p in range(result_df.shape[1])], rotation='vertical')
    #ax.set_xlabel('Position')
    #ax.set_ylabel('Percentage of correctly predicted sequence behaviors')
    ax.set_ylabel('Percentage of correct prediction')
    
    plt.xticks([])
    plt.tight_layout()

    plt.title(behav)
    #plt.show()
    #plt.legend()


plot_percentage(6,'seq')

plt.show()

