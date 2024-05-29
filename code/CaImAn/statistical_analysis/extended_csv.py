import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from scipy.stats import ttest_ind
from scipy.stats import norm
from scipy.stats import mannwhitneyu



def extract_sessions(name):
    csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
    df = pd.read_csv(csv_file_path)

    filtered_df = df[df.iloc[:,0].str.contains('FR1')]
    last_row_index = filtered_df.index[-1]

    new_list = []
    row_numbers = []

    for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
        new_list.append(elem)
        row_numbers.append(index)

    return row_numbers


big_name = 'I236'
folder = 'deconvoled'


data_df_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{big_name}/data_stat.csv'
original_data_df = pd.read_csv(data_df_path)






if big_name == 'I236':
    name_list = ['I2', 'I3', 'I6']
if big_name == 'L023':
    name_list = ['L0', 'L2', 'L3']

sc = 0
data_list = []
for name in name_list:

    row_numbers = extract_sessions(name)
    for session_index,session in enumerate(row_numbers):

        print(name, session)

        behaviors_list = ['sequence', 'zones', 'food', 'explo']

        freq_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/frequencies_4/{name}/freq_S{session+1}.csv'
        freq_df = pd.read_csv(freq_path, header=None)


        more_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/more_neurons/{name}_all/S{session+1}.csv'
        more_df = pd.read_csv(more_path, header=None)
        sc += len(more_df)

        for i,row in more_df.iterrows():
            row = row.tolist()
            index = row[1]
            cluster = row[2]

            freq_list = freq_df.iloc[index,:].tolist()
            
            for k in range(4):

                data_row = [index, behaviors_list[k], f'Cluster {cluster}', f'Session {session_index +1}', freq_list[k], name]
                data_list.append(data_row)



more_data_df = pd.DataFrame(data_list, columns=['Neuron_ID', 'Behavior', 'Cluster', 'Day_of_recording', 'Frequency', 'Name'])




data_df = pd.concat([original_data_df, more_data_df])

# Optionally, you can reset the index if you want a continuous index
data_df.reset_index(drop=True, inplace=True)


behav_list  = ['sequence', 'zones', 'food', 'explo']


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
    for cluster_num in range(1,7):
        for behav_ind, behav in enumerate(behav_list):
            for pair_ind,pair in enumerate(compare_list):

                cluster_1 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {pair[0]}') & (data_df['Behavior'] == behav)]
                cluster_2 = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {pair[1]}') & (data_df['Behavior'] == behav)]

                t_statistic, p_value = ttest_ind(cluster_1['Frequency'], cluster_2['Frequency'])
                #statistic, p_value = mannwhitneyu(cluster_1['Frequency'], cluster_2['Frequency'])


                mat[behav_ind + (cluster_num-1)*4, pair_ind] = p_value
                diff_mat[behav_ind + (cluster_num-1)*4, pair_ind] = cluster_2['Frequency'].mean() - cluster_1['Frequency'].mean()

        
    p_df = pd.DataFrame(mat)
    diff_df = pd.DataFrame(diff_mat)
    #print(p_df)
    print(diff_df)

    #p_df = p_df.map(lambda x: 1 if x <= 0.05 else 0)

    new_df = p_df.copy()
    new_df[p_df > 0.05] = 0
    mask_pos = (p_df < 0.05) & (diff_df > 0)
    mask_neg = (p_df < 0.05) & (diff_df < 0)

    new_df[mask_pos] = 1
    new_df[mask_neg] = -1

    p_df = new_df



    labels_y = behav_list 
    labels_x = ['FR1_1 -> FR1_4', 'FR1_4 -> FR5_1', 'FR5_1 -> FR5_4']
    fig, axes = plt.subplots(6, 1, figsize=(10, 7))
    for i, axi in enumerate(np.arange(0,6)):
        start_index = i * 4
        end_index = (i + 1) * 4
        subset_p_df = p_df.iloc[start_index:end_index]
        
        sns.heatmap(subset_p_df, cmap='coolwarm', annot=False, fmt=".2f", ax=axes[5-axi], cbar=False)
        #axes[5-axi].set_xticks([tick + 0.5 for tick in range(len(labels_x))])
        #axes[5-axi].set_xticklabels(labels_x)
        axes[5-axi].set_yticks([tick + 0.5 for tick in range(len(labels_y))])
        axes[5-axi].set_yticklabels(labels_y, rotation = 'horizontal')
        axes[5-axi].set_title(f'Cluster {i+1}')
        axes[5-axi].set_xticks([])
        axes[5-axi].set_xticklabels([])
    
    
    plt.tight_layout()


    plt.figure()
    sns.heatmap(p_df, cmap='coolwarm', annot=False, fmt=".2f")
    labels_x = ['FR1_1 -> FR1_4', 'FR1_4 -> FR5_1', 'FR5_1 -> FR5_4']
    labels_y = behav_list * 6
    plt.xticks([tick + 0.5 for tick in range(len(labels_x))], labels_x)
    plt.yticks([tick + 0.5 for tick in range(len(labels_y))], labels_y)

    plt.show()
#p_first_last()

def p_first_last_modif():

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
                diff_mat[behav_ind + (cluster_num-1)*4, pair_ind] = (cluster_2['Frequency'].mean() - cluster_1['Frequency'].mean())/cluster_1['Frequency'].mean()

                formatted_mean1 = "{:.3e}".format(cluster_1['Frequency'].mean())
                formatted_mean2 = "{:.3e}".format(cluster_2['Frequency'].mean())

                str = f"{formatted_mean1} -> {formatted_mean2}"
                #str = f"{(cluster_2['Frequency'].mean() - cluster_1['Frequency'].mean())/cluster_1['Frequency'].mean()}"
                val_mat[behav_ind + (cluster_num-1)*4, pair_ind] = str
        

        
    p_df = pd.DataFrame(mat)
    diff_df = pd.DataFrame(diff_mat)
    print(p_df)
    print(diff_df)

    val_df = pd.DataFrame(val_mat)

    print(val_df)

    #p_df = p_df.map(lambda x: 1 if x <= 0.05 else 0)

    new_diff_df = diff_df.copy()
    mask_p = (p_df > 0.05)

    new_diff_df[mask_p] = 0.0
    #new_df[mask_neg] = -1

    #p_df = new_df

    min = new_diff_df.min().min()
    max = new_diff_df.max().max()

    #print(new_diff_df)


    annot = val_df
    #annot = val_df.where(~mask_p, '')




    labels_y = behav_list 
    labels_x = ['FR1_1 -> FR1_4', 'FR1_4 -> FR5_1', 'FR5_1 -> FR5_4']
    fig, axes = plt.subplots(6, 1, figsize=(10, 7))
    for i, axi in enumerate(np.arange(0,6)):
        start_index = i * 4
        end_index = (i + 1) * 4
        subset_p_df = new_diff_df.iloc[start_index:end_index]
        sub_annot = annot.iloc[start_index:end_index]
        #sns.heatmap(subset_p_df, cmap='bwr', annot=False, fmt=".2f", ax=axes[5-axi], cbar=True, vmin=min, vmax=max)

        heatmap = sns.heatmap(subset_p_df, cmap='coolwarm', mask=subset_p_df == 0.0, cbar=True,ax=axes[5-axi], vmin=min,vmax=max, fmt='',annot=sub_annot)
        
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
        for text in heatmap.texts:
            if text.get_text() == '0.0':
                text.set_color('green')



        #axes[5-axi].set_xticks([tick + 0.5 for tick in range(len(labels_x))])
        #axes[5-axi].set_xticklabels(labels_x)
        axes[5-axi].set_yticks([tick + 0.5 for tick in range(len(labels_y))])
        axes[5-axi].set_yticklabels(labels_y, rotation = 'horizontal')
        axes[5-axi].set_title(f'Cluster {i+1}')
        axes[5-axi].set_xticks([])
        axes[5-axi].set_xticklabels([])
    
    
    plt.tight_layout()


    # plt.figure()
    # sns.heatmap(new_diff_df, cmap='coolwarm', annot=False, fmt=".2f")
    # labels_x = ['FR1_1 -> FR1_4', 'FR1_4 -> FR5_1', 'FR5_1 -> FR5_4']
    # labels_y = behav_list * 6
    # plt.xticks([tick + 0.5 for tick in range(len(labels_x))], labels_x)
    # plt.yticks([tick + 0.5 for tick in range(len(labels_y))], labels_y)



    # plt.figure()
    # sns.heatmap(new_diff_df, cmap='coolwarm', mask=new_diff_df == 0.0, cbar=False, vmin=min,vmax=max)
    # # Customize the color for zero values to be white
    # plt.cm.get_cmap('coolwarm').set_bad(color='lightgrey')


    plt.show()
p_first_last_modif()




# plt.figure()
# plt.bar(1,0,label='No significant diff.', color = 'lightgrey')
# plt.bar(1,1,label='Significant negative diff.', color ='mediumblue')
# plt.bar(1,-1,label='Significant positive diff.', color ='firebrick')
# plt.legend()
# plt.show()



def means():
    mat = np.zeros([24,8])
    diff_mat = np.zeros([24,8])
    

    for cluster_num in range(1,7):
        for behav_ind, behav in enumerate(behav_list):
            for i in range(1,9):

                cluster = data_df[(data_df['Cluster'] == f'Cluster {cluster_num}') & (data_df['Day_of_recording'] == f'Session {i}') & (data_df['Behavior'] == behav)]



                diff_mat[behav_ind + (cluster_num-1)*4, i-1] = cluster['Frequency'].mean()

                

    p_df = pd.DataFrame(mat)
    p_df = p_df.map(lambda x: 1 if x <= 0.05 else 0)


    diff_df = pd.DataFrame(diff_mat)

    min = diff_df.min().min()
    max = diff_df.max().max()

    print(min,max)


    labels_y = behav_list * 6
    labels_x = ['1', '2', '3', '4', '5', '6', '7', '8']
    sns.heatmap(diff_df, cmap='coolwarm', annot=False, fmt=".2f", vmin=min, vmax=10*min)
    plt.xticks([tick + 0.5 for tick in range(len(labels_x))], labels_x)
    plt.yticks([tick + 0.5 for tick in range(len(labels_y))], labels_y)




    labels_y = behav_list 
    fig, axes = plt.subplots(6, 1, figsize=(10, 7))
    for i, axi in enumerate(np.arange(0,6)):
        start_index = i * 4
        end_index = (i + 1) * 4
        subset_p_df = diff_df.iloc[start_index:end_index]

        min = subset_p_df.min().min()
        max = subset_p_df.max().max()
  
        
        sns.heatmap(subset_p_df, cmap='coolwarm', annot=True, fmt=".5f", ax=axes[5-axi],vmax=4*min)
        axes[5-axi].set_xticks([tick + 0.5 for tick in range(len(labels_x))])
        axes[5-axi].set_xticklabels(labels_x)
        axes[5-axi].set_yticks([tick + 0.5 for tick in range(len(labels_y))])
        axes[5-axi].set_yticklabels(labels_y, rotation = 'horizontal')
        axes[5-axi].set_title(f'Cluster {i+1}')

    plt.tight_layout()
    plt.show()

#means()