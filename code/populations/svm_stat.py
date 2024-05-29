import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

big_name = 'I236'



if big_name == 'I236':
    name_list = ['I2', 'I3', 'I6']
if big_name == 'L023':
    name_list = ['L0', 'L2', 'L3']



def get_p(type):

    path1 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{type}/{big_name}/decon_6_all/{name_list[0]}.csv'
    df1 = pd.read_csv(path1, header =None)

    path2 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{type}/{big_name}/decon_6_all/{name_list[1]}.csv'
    df2 = pd.read_csv(path2, header =None)

    path3 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{type}/{big_name}/decon_6_all/{name_list[2]}.csv'
    df3 = pd.read_csv(path3, header =None)


    anova_results = []

    for i in range(df1.shape[1]):
        data_group1 = df1.iloc[:,i]
        data_group2 = df2.iloc[:,i]
        data_group3 = df3.iloc[:,i]

        print(data_group2.tolist())

        anova_result = f_oneway(data_group1, data_group2, data_group3)

        if i == 1:
            anova_result = f_oneway(data_group1, data_group3)
        anova_results.append(anova_result.pvalue)

    #print(anova_results)
    return anova_results


def plot_p():
    type_list = ['global', 'seq', 'food', 'zone', 'explo']

    for type in type_list:

        result_list = get_p(type)
        #print(result_list)

        data_array = pd.DataFrame(result_list)
        data_array = data_array.T
        
        plt.figure(figsize=(14,1))
        sns.heatmap(data_array, annot=True, cmap="Greys", fmt=".2e", vmin=1e32, linewidths=2, linecolor='black')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()



def get_corr(type):

    path1 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{type}/{big_name}/decon_6_all/{name_list[0]}.csv'
    df1 = pd.read_csv(path1, header =None)
    path2 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{type}/{big_name}/decon_6_all/{name_list[1]}.csv'
    df2 = pd.read_csv(path2, header =None)
    path3 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/percentage_{type}/{big_name}/decon_6_all/{name_list[2]}.csv'
    df3 = pd.read_csv(path3, header =None)

    num_path1 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/size_sub/{big_name}/decon_6_all/num_{name_list[0]}.csv'
    num_df1 = pd.read_csv(num_path1, header =None)
    num_path2 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/size_sub/{big_name}/decon_6_all/num_{name_list[1]}.csv'
    num_df2 = pd.read_csv(num_path2, header =None)
    num_path3 = f'/Users/cyrilvanleer/Desktop/Thesis/populations/size_sub/{big_name}/decon_6_all/num_{name_list[2]}.csv'
    num_df3 = pd.read_csv(num_path3, header =None)


    fig, axs = plt.subplots(2, 3, figsize=(16, 4))

    #for i in range(df1.shape[1]):
    for i in range(df1.shape[0]):
        data_group1 = df1.iloc[i,:].tolist()
        data_group2 = df2.iloc[i,:].tolist()
        data_group3 = df3.iloc[i,:].tolist()


        data_list = data_group1 + data_group2 + data_group3

        num_group1 = num_df1.iloc[i,:].tolist()
        num_group2 = num_df2.iloc[i,:].tolist()
        num_group3 = num_df3.iloc[i,:].tolist()

        num_list = num_group1 + num_group2 + num_group3

        if big_name=='L023' and i==1:
            num_list = num_group1 + num_group3
            data_list = data_group1 +  data_group3
        

        row_index = i // 3
        col_index = i % 3 

        third = int(len(data_list)/3)
        marker_list = ['o', 'v', 'D']
        axs[row_index,col_index].scatter(num_list[0:third],data_list[0:third], color='none', edgecolor='blue')
        axs[row_index,col_index].scatter(num_list[third:2*third],data_list[third:2*third], color='none',  edgecolor='red')
        axs[row_index,col_index].scatter(num_list[2*third:3*third],data_list[2*third:3*third],color='none',  edgecolor='green')

        #axs[row_index,col_index].set_title(f'{np.corrcoef(num_list,data_list)[0, 1]}')

        if i==0:
            axs[row_index,col_index].set_xlabel('Subset size')
            axs[row_index,col_index].set_ylabel('Prediction quality')


    plt.tight_layout()
    plt.show()



get_corr('global')


# plt.figure()
# plt.scatter(1,1, label=name_list[0],color='none', edgecolor='blue')
# plt.scatter(1,1, label=name_list[1],color='none', edgecolor='red')
# plt.scatter(1,1, label=name_list[2],color='none', edgecolor='green')

# plt.legend()
# plt.show()