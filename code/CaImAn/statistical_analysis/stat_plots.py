import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_ind
from scipy.stats import norm
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.colors import Normalize
from scipy.stats import mannwhitneyu



from scipy.stats import shapiro




big_name = 'I236'
folder = 'deconvoled'


data_df_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{big_name}/data_stat.csv'
data_df = pd.read_csv(data_df_path)

#print(data_df)

behav_list  = ['sequence', 'zones', 'food', 'explo']

# Several plots of the data 
def plot():

    mean_freq = data_df['Frequency'].mean()
    std_freq = data_df['Frequency'].std()


    lower_bound = mean_freq - 3 * std_freq
    upper_bound = mean_freq + 3 * std_freq

    df_filtered = data_df[(data_df['Frequency'] >= lower_bound) & (data_df['Frequency'] <= upper_bound)]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_filtered, x='Day_of_recording', y='Frequency', hue='Behavior', palette='Set1')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_df, x='Day_of_recording', y='Frequency', hue='Behavior', palette='Set1')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_filtered, x='Day_of_recording', y='Frequency', hue='Behavior', palette='Set1')

    plt.title('Frequency vs. Day of Recording')
    plt.xlabel('Day of Recording')
    plt.ylabel('Frequency')

    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()

    plt.show()
#plot()

# Big histograms of all the frequencies recorded
def freq_distrib():
    plt.hist(data_df['Frequency'], bins=30, density=True, alpha=0.6, color='g', label='Histogram')

    # Fit a normal distribution to the data
    mu, std = norm.fit(data_df['Frequency'])

    # Plot the PDF of the normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution')

    plt.xlabel('Frequency')
    plt.ylabel('Density')
    plt.title('Histogram of Frequency with Fitted Normal Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
#freq_distrib()

# P values for 2 consecutives sessions
def p_consecutive():
    mat = np.zeros([24,7])
    diff_mat = np.zeros([24,7])

    for cluster_num in range(1,7):
        for behav_ind, behav in enumerate(behav_list):
            for i in range(1,8):
                
                cluster_1 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {i}') & (data_df['Behavior'] == behav)]
                cluster_2 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {i+1}') & (data_df['Behavior'] == behav)]

                t_statistic, p_value = ttest_ind(cluster_1['Frequency'], cluster_2['Frequency'])
                #statistic, p_value = mannwhitneyu(cluster_1['Frequency'], cluster_2['Frequency'])

                mat[behav_ind + (cluster_num-1)*4, i-1] = p_value

                diff_mat[behav_ind + (cluster_num-1)*4, i-1] = cluster_2['Frequency'].mean() - cluster_1['Frequency'].mean()


    p_df = pd.DataFrame(mat)
    p_df = p_df.map(lambda x: 1 if x <= 0.05 else 0)


    labels_y = behav_list * 6
    labels_x = ['1->2', '2->3', '3->4', '4->5', '5->6', '6->7', '7->8']
    sns.heatmap(p_df, cmap='coolwarm', annot=False, fmt=".2f")
    plt.xticks([tick + 0.5 for tick in range(len(labels_x))], labels_x)
    plt.yticks([tick + 0.5 for tick in range(len(labels_y))], labels_y)


    labels_y = behav_list 
    fig, axes = plt.subplots(6, 1, figsize=(10, 7))
    for i, ax in enumerate((axes.flat)):
        start_index = i * 4
        end_index = (i + 1) * 4
        subset_p_df = p_df.iloc[start_index:end_index]
        
        sns.heatmap(subset_p_df, cmap='coolwarm', annot=False, fmt=".2f", ax=ax)
        ax.set_xticks([tick + 0.5 for tick in range(len(labels_x))])
        ax.set_xticklabels(labels_x)
        ax.set_yticks([tick + 0.5 for tick in range(len(labels_y))])
        ax.set_yticklabels(labels_y, rotation = 'horizontal')
        ax.set_title(f'Cluster {i+1}')

    plt.tight_layout()
    #plt.show()


    diff_df = pd.DataFrame(diff_mat).mul(100000)

    labels_y = behav_list 
    fig, axes = plt.subplots(6, 1, figsize=(10, 7))
    for i, ax in enumerate((axes.flat)):
        start_index = i * 4
        end_index = (i + 1) * 4
        subset_p_df = diff_df.iloc[start_index:end_index]
        
        sns.heatmap(subset_p_df, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
        ax.set_xticks([tick + 0.5 for tick in range(len(labels_x))])
        ax.set_xticklabels(labels_x)
        ax.set_yticks([tick + 0.5 for tick in range(len(labels_y))])
        ax.set_yticklabels(labels_y, rotation = 'horizontal')
        ax.set_title(f'Cluster {i+1}')

    plt.tight_layout()

    plt.show()
#p_consecutive()

# P values between the extreme sessions of FR
def p_first_last():

    compare_list = [[1,4],[4,5],[5,8]]

    mat = np.zeros([24,3])
    diff_mat = np.zeros([24,3])
    val_mat = np.zeros([24,3], dtype=object)
    for cluster_num in range(1,7):
        for behav_ind, behav in enumerate(behav_list):
            for pair_ind,pair in enumerate(compare_list):

                cluster_1 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {pair[0]}') & (data_df['Behavior'] == behav)]
                cluster_2 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {pair[1]}') & (data_df['Behavior'] == behav)]

                t_statistic, p_value = ttest_ind(cluster_1['Frequency'], cluster_2['Frequency'])
                #statistic, p_value = mannwhitneyu(cluster_1['Frequency'], cluster_2['Frequency'])


                mat[behav_ind + (cluster_num-1)*4, pair_ind] = p_value
                diff_mat[behav_ind + (cluster_num-1)*4, pair_ind] = cluster_1['Frequency'].mean() - cluster_1['Frequency'].mean()

                str = f"{cluster_1['Frequency'].mean()} -> {cluster_1['Frequency'].mean()}"
                val_mat[behav_ind + (cluster_num-1)*4, pair_ind] = str
        
    p_df = pd.DataFrame(mat)
    diff_df = pd.DataFrame(diff_mat)

    val_df = pd.DataFrame(val_mat)

    print(val_df)

    #p_df = p_df.map(lambda x: 1 if x <= 0.05 else 0)

    new_df = p_df.copy()
    new_df[p_df > 0.05] = 0
    mask_pos = (p_df < 0.05) & (diff_df > 0)
    mask_neg = (p_df < 0.05) & (diff_df < 0)

    new_df[mask_pos] = 1
    new_df[mask_neg] = -1

    p_df = new_df


    sns.heatmap(p_df, cmap='seismic', annot=False, fmt=".2f")
    labels_x = ['FR1_1 -> FR1_4', 'FR1_4 -> FR5_1', 'FR5_1 -> FR5_4']
    labels_y = behav_list * 6
    plt.xticks([tick + 0.5 for tick in range(len(labels_x))], labels_x)
    plt.yticks([tick + 0.5 for tick in range(len(labels_y))], labels_y)

    plt.show()
#p_first_last()

# Plot to see if the values of the p values evolve with the sessions        USELESS
def evol_p_values():
    mat = np.zeros([24,7])
    for cluster_num in range(1,7):
        for behav_ind, behav in enumerate(behav_list):
            for i in range(1,8):
                
                cluster_1 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {i}') & (data_df['Behavior'] == behav)]
                cluster_2 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {i+1}') & (data_df['Behavior'] == behav)]

                t_statistic, p_value = ttest_ind(cluster_1['Frequency'], cluster_2['Frequency'])

                mat[behav_ind + (cluster_num-1)*4, i-1] = p_value


    p_df = pd.DataFrame(mat)

    x_ticks = []
    cmap = matplotlib.colormaps['Purples']


    plt.figure()
    x_values = [1 + k * 4 for k in range(len(p_df) // 4)]
    for i in range(len(p_df)):
        x_offset = -0.75 + (i % 4) * 0.5  
        x = x_values[i // 4] + x_offset
        x_ticks.append(x)    

        y = p_df.iloc[i].values  
        for j, val in enumerate(y):
            color = cmap(j / len(y))  # Use column index to determine color
            plt.scatter(x, val, color=color)


    plt.xticks(x_ticks, ['seq', 'zones', 'food', 'explo']*6, rotation ='vertical' )
    plt.axhline(y=0.05, color='red', linestyle='--')

    norm = Normalize(vmin=0, vmax=6)  # Assuming your data values range from 0 to 6
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar
    cax = plt.gca().inset_axes([1.0, 0, 0.05, 1])  # Adjust position and size as needed
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Session transition number')

    plt.xlabel('X')
    plt.ylabel('Value')
    plt.title('P values of consecutive sessions')
    plt.show()
#evol_p_values()

# Scatter/Box plot of two specific subset (used to highlight significant differences or not)
def distrib():

    sessions = [1,4]
    cluster_num = 1
    behav = 'sequence'


    cluster_1 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {sessions[0]}') & (data_df['Behavior'] == behav)]
    cluster_2 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {sessions[1]}') & (data_df['Behavior'] == behav)]


    list_1 = cluster_1.iloc[:,4].tolist()
    list_2 = cluster_2.iloc[:,4].tolist()

    for elem in list_1:
        plt.scatter(0,elem, color = 'white', edgecolors='black')
    for elem in list_2:
        plt.scatter(1,elem, color = 'white', edgecolors='black')

    plt.boxplot([list_1, list_2], positions=[0, 1], showmeans=True, meanline=False, widths=0.6, labels=['Subset 1', 'Subset 2'])

    plt.show()
#distrib()

# Histograms of the distributions of the frequencies (useless for now)
def histograms():
    #data_df = data_df[data_df['Cluster'] == 'Cluster 1']

    min_freq = data_df['Frequency'].min()
    max_freq = data_df['Frequency'].max()

    fig, axes = plt.subplots(nrows=len(data_df['Behavior'].unique()), 
                            ncols=len(data_df['Day_of_recording'].unique()), 
                            figsize=(12, 8))

    # Iterate over unique values of 'Behavior'
    for i, behavior_value in enumerate(data_df['Behavior'].unique()):
        # Subset the DataFrame by 'Behavior'
        subset_behavior = data_df[data_df['Behavior'] == behavior_value]
        
        # Iterate over unique values of 'Day_of_recording'
        for j, day_value in enumerate(data_df['Day_of_recording'].unique()):
            # Subset the DataFrame by both 'Behavior' and 'Day_of_recording'
            subset = subset_behavior[subset_behavior['Day_of_recording'] == day_value]
            
            for cluster_value in subset['Cluster'].unique():
                # Subset the DataFrame by 'Cluster'
                cluster_subset = subset[subset['Cluster'] == cluster_value]
                
                # Plot histogram for the current cluster
                axes[i, j].hist(cluster_subset['Frequency'], bins=10, alpha=0.5, density=True, range=(0, 0.0002))

                stat, p = shapiro(cluster_subset['Frequency'].tolist())
                print(behavior_value, day_value, cluster_value, p)
                    




    # Adjust layout
    plt.tight_layout()
    plt.show()
#histograms()


# mat = np.zeros([4,8])

# for behav_ind, behav in enumerate(behav_list):
#     for j in range(1,9):

#         subsets = [data_df[(data_df['Behavior'] == behav) & (data_df['Cluster'] == f'Cluster {i}') & (data_df['Day_of_recording'] == f'Session {j}')] for i in range(1, 7)]

#         statistic, p_value = f_oneway(*[subset['Frequency'] for subset in subsets])

#         mat[behav_ind, j-1] = p_value

# p_df = pd.DataFrame(mat)

# p_df.index = ['sequence', 'zones', 'food', 'explo']
# p_df.columns = [ f'Session {i}' for i in range(1,9)]

# print(p_df)