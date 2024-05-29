import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm, ols

from scipy.stats import norm



red_colors = ['mistyrose', 'lightcoral', 'indianred', 'brown', 'firebrick', 'darkred', 'maroon', 'black']
blue_colors = ['lightcyan', 'lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'darkblue', 'midnightblue']
green_colors = ['honeydew', 'lightgreen', 'palegreen', 'mediumseagreen', 'springgreen', 'limegreen', 'forestgreen', 'darkgreen']
purple_colors = ['lavender', 'thistle', 'plum', 'mediumorchid', 'darkorchid', 'indigo', 'mediumslateblue', 'darkviolet']


colors = red_colors + blue_colors + green_colors + purple_colors
markers_list = ['o', 's', '+', 'x', '*', 'D']

behaviors_list = ['sequence', 'zones', 'food', 'explo']

big_name = 'L023'
folder = 'deconvoled'


additional = False




if big_name == 'L023':
    class_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{big_name}/neuron_class_mouse_6.csv'
if big_name == 'I236':
    class_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{big_name}/neuron_class_mouse_6_average.csv'
df_class = pd.read_csv(class_path, header=None)



def get_lr_path(name):

    list = []

    list.append(['L0', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L0/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['L2', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L2/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['L3', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L3/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['L4', '/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/FR5 CUED/Longitudinal fr1-fr5cued L4/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['P4','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_a2ap4/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['D1_1','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/1 triss/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['D1_2','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/2 phi/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['D1_3','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2022_9_D1/longit/3 tis/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['I2','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i2/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['I3','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i3/Extracted_Data/Longitudinal_Registration.csv'])
    list.append(['I6','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2021_5_2021_7_D1_A2A/longit/longit_d1i6/Extracted_Data/Longitudinal_Registration.csv'])

    lr_path_df = pd.DataFrame(list)
    long_reg_path = lr_path_df[lr_path_df[0] == name].iloc[0,1]

    return long_reg_path

def cut_lf_df(long_reg_df):
    if name == 'I2':
        long_reg_df = long_reg_df.iloc[:100,:]
    if name == 'I3':
        long_reg_df = long_reg_df.iloc[:107,:]
        long_reg_df = long_reg_df.reset_index(drop=True)
    if name == 'I6':
        long_reg_df = long_reg_df.iloc[:101,:]
        long_reg_df = long_reg_df.reset_index(drop=True)


    if name == 'L0':
        long_reg_df = long_reg_df.iloc[:100,:]
    if name == 'L2':
        long_reg_df = long_reg_df.iloc[:50,:]
        long_reg_df = long_reg_df.reset_index(drop=True)
    if name == 'L3':
        long_reg_df = long_reg_df.iloc[:100,:]
        long_reg_df = long_reg_df.reset_index(drop=True)

    return long_reg_df

def slide_index(filter_list):
    sub_long = []
    for elem in filter_list:
        if name == 'I2':
            sub_long.append(long_reg_df.iloc[elem,:])
        if name == 'I3':
            sub_long.append(long_reg_df.iloc[elem-100,:])
        if name == 'I6':
            sub_long.append(long_reg_df.iloc[elem-207,:])


        if name == 'L0':
            sub_long.append(long_reg_df.iloc[elem,:])
        if name == 'L2':
            sub_long.append(long_reg_df.iloc[elem-100,:])
        if name == 'L3':
            sub_long.append(long_reg_df.iloc[elem-150,:])


    sub_df = pd.DataFrame(sub_long)
    return sub_df

if big_name == 'I236':
    name_list = ['I2', 'I3', 'I6']
if big_name == 'L023':
    name_list = ['L0', 'L2', 'L3']

data_list = []
    
fig, axs = plt.subplots(6, 1, figsize=(10, 7))


for name in name_list:

    print( '####### \n#### \n#######')
    

    long_reg_path = get_lr_path(name)
    long_reg_df = pd.read_csv(long_reg_path, header=None)
    long_reg_df = cut_lf_df(long_reg_df)

    neurons_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'
    df = pd.read_csv(neurons_path)

    filtered_df = df[df.iloc[:,0].str.contains('FR1')]
    last_row_index = filtered_df.index[-1]

    new_list = []
    row_numbers = []

    for index, elem in df.iloc[[0, 1, last_row_index - 1, last_row_index, last_row_index + 1, last_row_index + 2, -2, -1]].iterrows():
        new_list.append(elem)
        row_numbers.append(index)

    print(row_numbers)

    ## get 4 freq for each neuron for the 8 sessions

    freq_dfs = {f'freq_df{i}': pd.read_csv(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/frequencies_4/{name}/freq_S{row+1}.csv', header=None) for i,row in enumerate(row_numbers)}

    #print(freq_dfs[f'freq_df{0}'])



    for session_index, session in enumerate(row_numbers):

        for i in range(1,7):

            sub = df_class[(df_class[1] == i) & (df_class[2] == name)]   


            filter_list = sub.iloc[:,0].tolist()
            sub_df = slide_index(filter_list)

            # print(filter_list)
            # print(sub_df)

            ########

            if name == 'I6':
                sub_df = sub_df.drop(sub_df.columns[2], axis=1)
                sub_df.columns = range(len(sub_df.columns))

            ########

            #print(name, session, i )

            sub_freq_list = []

            for j in range(len(sub_df)):
                index = sub_df.iloc[j,session]
                if index != 0:
                    index = index -1

                    #print(sub_df.index[j])


                    for k in range(4):
                        #data_row = [sub_df.index[j], k, i, session_index +1, freq_dfs[f'freq_df{session_index}'].iloc[index,k]]


                        data_row = [sub_df.index[j], behaviors_list[k], f'Cluster {i}', f'Session {session_index +1}', freq_dfs[f'freq_df{session_index}'].iloc[index,k], name]

                        if name == 'I3':
                            data_row[0] += 100
                        if name == 'I6':
                            data_row[0] += 207

                        if name == 'L2':
                            data_row[0] += 100
                        if name == 'L3':
                            data_row[0] += 150
                        



                        data_list.append(data_row)
                
            if additional == True:
                more_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/more_neurons/{name}_all/S{session+1}.csv'
                more_df = pd.read_csv(more_path,header=None)


                for i,row, in more_df.iterrows():
                    row = row.tolist()

                    index = row[1]
                    cluster = row[2]
                    print(index,cluster)




            #         sub_freq_list.append(freq_dfs[f'freq_df{session_index}'].iloc[index,:].tolist())


            # sub_freq_df = pd.DataFrame(sub_freq_list)

            # for k in range(4):
            #     y = sub_freq_df.iloc[:,k].tolist()
            #     if y != 50.0:
            #         axs[i-1].scatter([k +6*session_index] * len(y), y, c= colors[session_index +8*k], label=f'{session} {i} {k}')
            #         axs[i-1].set_ylim(0, 0.0003)
            #         #plt.scatter([k +6*session_index] * len(y), y, c= colors[session_index +8*k], marker = markers_list[i-1], label=f'{session} {i} {k}')


                
#plt.show()



data_df = pd.DataFrame(data_list, columns=['Neuron_ID', 'Behavior', 'Cluster', 'Day_of_recording', 'Frequency', 'Name'])



data_df['Frequency'].replace(50, 0.0, inplace=True)

#data_df.to_csv(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/similarities_8/{big_name}/data_stat.csv', index=False, header=True)

# print(data_df)
# print(data_df[data_df['Neuron_ID'] == 0])






def lmm_analysis():


    formula = 'Frequency ~  Behavior  + Cluster + Day_of_recording + Name'

    lmm = mixedlm(formula, data_df, groups=data_df['Neuron_ID'])
    lmm_fit = lmm.fit()

    print(lmm_fit.summary())

    print('Precise P>|z|)')
    print(lmm_fit.pvalues)

    # Extract the fixed effects (intercepts and coefficients) from the LMM
    fixed_effects = lmm_fit.fe_params

    # Define your ANOVA model using the fixed effects from the LMM
    anova_formula = 'Frequency ~ Behavior + Cluster + Day_of_recording + Name'
    anova_model = ols(anova_formula, data_df).fit()

    anova_table = sm.stats.anova_lm(anova_model)
    print(anova_table)








lmm_analysis()

