import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import csv


def find_behavior(df):

    list = []
    for i, row in df.iterrows():
    
        position = []
        positions_lower_5 = []
        positions_greater_95 = []
        
        for x, value in enumerate(row):
            if value < 5:
                positions_lower_5.append(x)
            elif value > 95:
                positions_greater_95.append(x)
        
        if positions_lower_5 and positions_greater_95:
            closest_to_0_idx = min(positions_lower_5, key=lambda x: abs(row[x]))
            closest_to_100_idx = min(positions_greater_95, key=lambda x: abs(row[x] - 100))
            
            closest_to_0_diff = abs(row[closest_to_0_idx])
            closest_to_100_diff = abs(row[closest_to_100_idx] - 100)
            
            if closest_to_0_diff < closest_to_100_diff:
                position.append(closest_to_0_idx)
            if closest_to_0_diff > closest_to_100_diff:
                position.append(closest_to_100_idx)

        if positions_lower_5 and not positions_greater_95:
            closest_to_0_idx = min(positions_lower_5, key=lambda x: abs(row[x]))
            position.append(closest_to_0_idx)

        if not positions_lower_5 and positions_greater_95:
            closest_to_100_idx = min(positions_greater_95, key=lambda x: abs(row[x] - 100))
            position.append(closest_to_100_idx)

        if not position:
            closest_to_5 = min(range(len(row)), key=lambda i: abs(row[i] - 5))
            closest_to_95 = min(range(len(row)), key=lambda i: abs(row[i] - 95))
            position.append(closest_to_5 if abs(row[closest_to_5] - 5) < abs(row[closest_to_95] - 95) else closest_to_95)
            
        #print(f"Row {i}:", ', '.join(map(str, row)), "Positions:", positions)

        list.extend(position)
    return list

def find_behaviors(df):

    list = []
    sum_list = [0,0,0,0,0,0,0]
    for i, row in df.iterrows():
    
        positions = [0, 0, 0, 0, 0, 0, 0]
        for x, value in enumerate(row):
            if value < 5:
                positions[x] = 1
                sum_list[x] += 1
            elif value > 95:
                positions[x] = 1
                sum_list[x] += 1

        list.append(positions)

    return list, sum_list


def df_behav():
    behav_list = []
    active_behav_list = []
    silent_behav_list = []
    for i in range(1,9):
        
        path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/L0/S{i}.csv'

        df = pd.read_csv(path, header=None)
        behav = find_behavior(df)
        category_counts = [behav.count(x) for x in range(7)]
        behav_list.append(category_counts)

        list_active = [0,0,0,0,0,0,0]
        list_silent = [0,0,0,0,0,0,0]


        for x, value in enumerate(behav):

            perc = df.iloc[x,value]
            if perc >= 50:
                list_active[value] += 1
            else:
                list_silent[value] += 1

        active_behav_list.append(list_active)
        silent_behav_list.append(list_silent)

    return behav_list, active_behav_list, silent_behav_list

def df_behavs():
    behav_list = []
    active_behav_list = []
    silent_behav_list = []
    ns_behav_list = []
    for i in range(1,9):
        
        path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/L0/S{i}.csv'

        df = pd.read_csv(path, header=None)
        behavs, category_counts = find_behaviors(df)
        behav_list.append(category_counts)

        list_active = [0,0,0,0,0,0,0]
        list_silent = [0,0,0,0,0,0,0]
        list_ns = [0,0,0,0,0,0,0]



        for x, row in enumerate(behavs):
            for y, value in enumerate(row):
                
                if value ==1:
                    perc = df.iloc[x,y]
                    if perc >= 95:
                        list_active[y] += 1
                    elif perc <= 5:
                        list_silent[y] += 1

                else:
                    list_ns[y] += 1 
        
        active_behav_list.append(list_active)
        silent_behav_list.append(list_silent)
        ns_behav_list.append(list_ns)

    return behav_list, active_behav_list, silent_behav_list, ns_behav_list


def df_behav_normalized():
    behav_list = []
    active_behav_list = []
    silent_behav_list = []
    for i in range(1,9):
        
        path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/L0/S{i}.csv'

        df = pd.read_csv(path, header=None)
        behav = find_behavior(df)

        category_counts = [behav.count(x) for x in range(7)]
        category_counts = [x / sum(category_counts) for x in category_counts]

        behav_list.append(category_counts)

        list_active = [0,0,0,0,0,0,0]
        list_silent = [0,0,0,0,0,0,0]


        for x, value in enumerate(behav):

            perc = df.iloc[x,value]
            if perc >= 50:
                list_active[value] += 1
            else:
                list_silent[value] += 1

        list_active = [x / sum(list_active) for x in list_active]
        list_silent = [x / sum(list_silent) for x in list_silent]

        active_behav_list.append(list_active)
        silent_behav_list.append(list_silent)
   
    return behav_list, active_behav_list, silent_behav_list

def df_behavs_normalized():
    behav_list = []
    active_behav_list = []
    silent_behav_list = []
    ns_behav_list = []
    for i in range(1,14):
        
        path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/I6/S{i}.csv'

        df = pd.read_csv(path, header=None)
        behavs, category_counts = find_behaviors(df)

    
        category_counts = [0 if x == 0 else x / sum(category_counts) for x in category_counts]
        behav_list.append(category_counts)

        list_active = [0,0,0,0,0,0,0]
        list_silent = [0,0,0,0,0,0,0]
        list_ns = [0,0,0,0,0,0,0]



        for x, row in enumerate(behavs):
            for y, value in enumerate(row):
                
                if value ==1:
                    perc = df.iloc[x,y]
                    if perc >= 95:
                        list_active[y] += 1
                    elif perc <= 5:
                        list_silent[y] += 1

                else:
                    list_ns[y] += 1 
            
        list_active = [0 if x == 0 else x / sum(list_active) for x in list_active]
        list_silent = [0 if x == 0 else x / sum(list_silent) for x in list_silent]
        
        active_behav_list.append(list_active)
        silent_behav_list.append(list_silent)
        ns_behav_list.append(list_ns)

    return behav_list, active_behav_list, silent_behav_list, ns_behav_list


def plot_results(behav_list, active_behav_list, silent_behav_list):
    num_categories = len(behav_list[0])
    bar_width = 0.05
    bar_positions = np.arange(num_categories)

      
    plt.style.use('ggplot')
    colors = ['darksalmon', 'lightcoral', 'indianred', 'brown', 'skyblue', 'cornflowerblue', 'royalblue', 'blue']
    colors_19 = ['mistyrose','pink','lightpink','lightsalmon','darksalmon', 'lightcoral','indianred','orangered', 'red', 'brown', 'darkred', 
    'skyblue','deepskyblue', 'cornflowerblue', 'royalblue', 'blue', 'mediumblue', 'darkblue', 'navy']

    #colors_19 = [(1.0, 1.0, 1.0), (1.0, 0.75, 0.75), (1.0, 0.5, 0.5), (1.0, 0.25, 0.25), (1.0, 0.0, 0.0), (0.8333333333333334, 0.0, 0.0), (0.6666666666666666, 0.0, 0.0), (0.5, 0.0, 0.0), (0.3333333333333333, 0.0, 0.0), (0.16666666666666666, 0.0, 0.0), (0.0, 0.0, 0.0),
    #(1.0, 1.0, 1.0), (0.6666666666666666, 0.6666666666666666, 1.0), (0.3333333333333333, 0.3333333333333333, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.75), (0.0, 0.0, 0.5), (0.0, 0.0, 0.25), (0.0, 0.0, 0.0)]
    
    colors_13 = ['mistyrose','pink','lightpink','indianred','orangered', 'red', 'brown', 'darkred', 
    'deepskyblue', 'cornflowerblue', 'royalblue', 'mediumblue', 'darkblue']

    colors_12 = ['mistyrose','pink','lightpink','indianred','orangered', 'red', 'brown', 
    'deepskyblue', 'cornflowerblue', 'royalblue', 'mediumblue', 'darkblue']

    colors_17 = ['mistyrose','pink','lightpink','lightsalmon','darksalmon', 'lightcoral','indianred','orangered', 'red', 'brown', 'darkred','black', 
    'skyblue', 'cornflowerblue', 'blue', 'mediumblue', 'navy']

    colors_13_D1_2 = ['mistyrose','pink','lightpink','darksalmon','indianred','orangered', 'red', 'brown', 'darkred', 
     'cornflowerblue', 'royalblue', 'mediumblue', 'darkblue']

    colors_12_I2 = ['mistyrose','lightpink','indianred','orangered', 'red', 'brown', 
    'skyblue','deepskyblue', 'cornflowerblue', 'royalblue', 'mediumblue', 'darkblue']

    colors_16_I3 = ['pink','lightpink','lightsalmon','darksalmon', 'lightcoral','indianred','orangered', 'red', 'brown', 'darkred', 
    'deepskyblue', 'cornflowerblue', 'royalblue', 'blue', 'mediumblue', 'darkblue']


    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for i, category_data in enumerate(behav_list):
        axs[0].bar(bar_positions + i * bar_width, category_data, width=bar_width, label=f'Data {i+1}', color=colors_13[i])
    axs[0].set_xticks(bar_positions + (len(behav_list) - 1) * bar_width / 2)
    axs[0].set_xticklabels(['sequence', 'zone1', 'zone2', 'pellet', 'no pellet', 'explo1', 'explo2'])
    axs[0].set_title('Significative neurons')

    for i, category_data in enumerate(active_behav_list):
        axs[1].bar(bar_positions + i * bar_width, category_data, width=bar_width, label=f'Data {i+1}', color=colors_13[i])
    axs[1].set_xticks(bar_positions + (len(behav_list) - 1) * bar_width / 2)
    axs[1].set_xticklabels(['sequence', 'zone1', 'zone2', 'pellet', 'no pellet', 'explo1', 'explo2'])
    axs[1].set_title('Active neurons')

    for i, category_data in enumerate(silent_behav_list):
        axs[2].bar(bar_positions + i * bar_width, category_data, width=bar_width, label=f'Data {i+1}', color=colors_13[i])
    axs[2].set_xticks(bar_positions + (len(behav_list) - 1) * bar_width / 2)
    axs[2].set_xticklabels(['sequence', 'zone1', 'zone2', 'pellet', 'no pellet', 'explo1', 'explo2'])
    axs[2].set_title('Silent neurons')


    plt.tight_layout()
    plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/neurons/images/behavior_from_percentile/I6.png' , dpi=300)
      


behavs_list_n, active_behavs_list_n, silent_behavs_list_n, ns_behavs_list_n = df_behavs_normalized()
plot_results(behavs_list_n, active_behavs_list_n, silent_behavs_list_n)
plt.show()


