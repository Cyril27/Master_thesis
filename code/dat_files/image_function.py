import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


column_names = ['time','family', 'num', 'P', 'V', 'L', 'R', 'T','W', 'X', 'Y', 'Z']


def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t', skiprows=12, header=None, names=column_names)
    return data

def enter_zone2(data, start):
    result_df = pd.DataFrame(columns=data.columns)

    for i in range(len(start)):
        x = start.iloc[i,0]
        lim = data[data.iloc[:, 0] == x].index[0]
        subset = data.iloc[lim:, :]

        # Find the first element after the specified index where the second column is equal to 2
        condition = (subset.iloc[:, 1] == 9) & (subset.iloc[:, 4] == 2) & (subset.index > lim)
        first_element_after = subset[condition].iloc[0] if not subset[condition].empty else None


        if first_element_after is not None:
            result_df = pd.concat([result_df, first_element_after.to_frame().T], ignore_index=True)

    result_df = result_df.drop_duplicates()

    
    return result_df


def enter_zone1(data, end):

    start=0
    result_df = pd.DataFrame(columns=data.columns)

    for i in range(len(end)):

        x = end.iloc[i,0]
        lim = data[data.iloc[:, 0] == x].index[0]
        
        subset=data.iloc[start:lim,:] 
        filtered_subset = subset[(subset.iloc[:, 1] == 9) & (subset.iloc[:, 4] == 1)]
        # Check if the filtered subset is not empty before accessing the last element
        if not filtered_subset.empty:
            last_element_y = filtered_subset.iloc[-1]
            result_df = pd.concat([result_df, last_element_y.to_frame().T], ignore_index=True)
        start = lim

    result_df = result_df.drop_duplicates()  

    
    return result_df



def start_end(data):

    subset = data[(data.iloc[:, 1] == 2)]



    end_df = pd.DataFrame(columns=data.columns)
    start_df = pd.DataFrame(columns=data.columns)
    
    if len(subset) != 0:

        for i in range(1, max(subset.iloc[:,7]) + 1):

            sub_subset = subset[(subset.iloc[:, 7] == i)]

            if len(sub_subset) > 1:
                start = sub_subset.iloc[0,:]
                end = sub_subset.iloc[1,:]



                end_df = pd.concat([end_df, end.to_frame().T], ignore_index=True)
                start_df = pd.concat([start_df, start.to_frame().T], ignore_index=True)

    return start_df,end_df


def get_zone(data, index_V=3):

    if index_V == 3:
        subset = data[data.iloc[:, 1] == 9]
    else:
        subset = data[(data.iloc[:, 1] == 9) & (data.iloc[:, 4] == index_V)]
        
    time_val = subset.iloc[:, 0]

    time_val = time_val.drop_duplicates()
    return time_val


def new_sequence(data):
    start, end = start_end(data)



    end_df = pd.DataFrame(columns=data.columns)
    start_df = pd.DataFrame(columns=data.columns)

    z1 = get_zone(data,1)
    z2 = get_zone(data,2)


    if len(z1) != 0:
        l = z1.iloc[0]
        #z2 = z2[z2 > l]
        #z2 = z2[:len(z1)]



        z2_new = []
        lim_inf = 0
        for i in range(1,len(z1)):
            lim_sup = z1.iloc[i]
            #print(lim_sup)
            sub_z2 = z2[(z2 >= lim_inf) & (z2 <= lim_sup)]
            #print(sub_z2)
            lim_inf = lim_sup

            if len(sub_z2 != 0):
                z2_new.append(sub_z2.iloc[-1])
            
        z2 = pd.Series(z2_new)

        for lim_1, lim_2 in zip(z1,z2):

            subset_start = start[(start.iloc[:, 0] <= lim_2) & (start.iloc[:, 0] >= lim_1)]    
            subset_end = end[(end.iloc[:, 0] <= lim_2) & (end.iloc[:, 0] >= lim_1)]  



            if  len(subset_start) != 0 :
                start_line = subset_start.iloc[0,:]
                start_df = pd.concat([start_df, start_line.to_frame().T], ignore_index=True)
            if len(subset_end) != 0 :
                end_line = subset_end.iloc[-1,:]
                end_df = pd.concat([end_df, end_line.to_frame().T], ignore_index=True)

        start_df = start_df.drop_duplicates()
        end_df = end_df.drop_duplicates()

        #print(len(start_df))
        end_df = end_df.iloc[:len(start_df),:]

        
    return start_df,end_df 

def get_time_pellet(data, index_P=3):

    if index_P == 3:
        subset = data[(data.iloc[:, 1] == 4) & (data.iloc[:, 4] == 1)]
    else:
        subset = data[(data.iloc[:, 1] == 4) & (data.iloc[:, 4] == 1) & (data.iloc[:, 3] == index_P)]

    time_val = subset.iloc[:, 0]
    return time_val

def pellet_interval(data,e1,e2):
    start_df = pd.DataFrame(columns=data.columns)
    end_df = pd.DataFrame(columns=data.columns)
    for x,y in zip(e2,e1):

        
        start = data[data.iloc[:, 0] == x].index[0]
        end = data[data.iloc[:, 0] == y].index[0]


        
        subset = data.iloc[start+1:end,:]
        #to be sure that there is no composed behavior

        if any((subset.iloc[:, 1] == 4) & (subset.iloc[:, 4] == 1) & (subset.iloc[:, 3] == 1)):

            element_start = data.iloc[start,:]
            start_df = pd.concat([start_df, element_start.to_frame().T], ignore_index=True)

            element_end = data.iloc[end,:]
            end_df = pd.concat([end_df, element_end.to_frame().T], ignore_index=True)

    return start_df,end_df

def no_pellet_interval(data,e1,e2):
    start_df = pd.DataFrame(columns=data.columns)
    end_df = pd.DataFrame(columns=data.columns)
    for x,y in zip(e2,e1):

        
        start = data[data.iloc[:, 0] == x].index[0]
        end = data[data.iloc[:, 0] == y].index[0]


        
        subset = data.iloc[start+1:end,:]


        if any((subset.iloc[:, 1] == 4) & (subset.iloc[:, 4] == 1) & (subset.iloc[:, 3] == 0)) and not any((subset.iloc[:, 1] == 4) & (subset.iloc[:, 4] == 1) & (subset.iloc[:, 3] == 1)) :
            

            element_start = data.iloc[start,:]
            start_df = pd.concat([start_df, element_start.to_frame().T], ignore_index=True)

            element_end = data.iloc[end,:]
            end_df = pd.concat([end_df, element_end.to_frame().T], ignore_index=True)

    return start_df,end_df

def explo_2(data,e1,e2):
    start_df = pd.DataFrame(columns=data.columns)
    end_df = pd.DataFrame(columns=data.columns)
    for x,y in zip(e2,e1):

        
        start = data[data.iloc[:, 0] == x].index[0]
        end = data[data.iloc[:, 0] == y].index[0]


        
        subset = data.iloc[start+1:end,:]

        if not any((subset.iloc[:, 1] == 4) & (subset.iloc[:, 4] == 1)) :
            

            element_start = data.iloc[start,:]
            start_df = pd.concat([start_df, element_start.to_frame().T], ignore_index=True)

            element_end = data.iloc[end,:]
            end_df = pd.concat([end_df, element_end.to_frame().T], ignore_index=True)

    return start_df,end_df


def get_z1_z2(data):
    start, end = start_end(data)
    z1 = get_zone(data,1)
    z2 = get_zone(data,2)

    z1_list = []
    z2_list = []

    for element in z1:
        #print(element)

        sub_2 = z2[(z2 >= element)]
        #print(sub_2)

        if len(sub_2) != 0:
            selected = sub_2.iloc[0]

            if selected not in z2_list:
                

                sub_1 = z1[(z1 <= selected) & (z1 >= element)]
                selected_1 = sub_1.iloc[0]

                
                z2_list.append(selected)
                z1_list.append(selected_1)




    z1_df = pd.Series(z1_list)
    z2_df = pd.Series(z2_list)
    return z1_df,z2_df



def explo_1(data):
    start, end = start_end(data)
    z1 = get_zone(data,1)
    z2 = get_zone(data,2)

    z1_list = []
    z2_list = []

    for element in z1:
        #print(element)

        sub_2 = z2[(z2 >= element)]
        #print(sub_2)

        if len(sub_2) != 0:
            selected = sub_2.iloc[0]

            if selected not in z2_list:
                

                sub_1 = z1[(z1 <= selected) & (z1 >= element)]
                selected_1 = sub_1.iloc[0]

                empty_sub = data[(data.iloc[:, 0] <= selected) & (data.iloc[:, 0] >= selected_1)]   
                if not any((empty_sub.iloc[:, 1] == 4) & (empty_sub.iloc[:, 4] == 1)) and not any((empty_sub.iloc[:, 1] == 2) & (empty_sub.iloc[:, 4] == 1)) :

                    z2_list.append(selected)
                    z1_list.append(selected_1)

            
    z1_df = pd.Series(z1_list)
    z2_df = pd.Series(z2_list)

    return z1_df,z2_df




def save_image(file_path):

    print(file_path)

    data = load_data(file_path)

    fig = plt.figure(figsize=(12, 2))

    start_action,end_action = start_end(data)
    start_action_time, end_action_time = start_action.iloc[:,0], end_action.iloc[:,0]
    plt.barh(y=1.25, width=np.array(end_action_time) - np.array(start_action_time), left=start_action_time, height=0.5, color="black", edgecolor='black', label='action')

    start_seq, end_seq = new_sequence(data)
    start_seq_time, end_seq_time = start_seq.iloc[:,0], end_seq.iloc[:,0]
    

    plt.barh(y=.75, width=np.array(end_seq_time) - np.array(start_seq_time), left=start_seq_time, height=0.5, color="blue", edgecolor='black', label='action sequence')


    end_zone2 = enter_zone2(data,end_seq)
    end_zone2_time = end_zone2.iloc[:,0]
    plt.barh(y=0.25, width=np.array(end_zone2_time) - np.array(end_seq_time), left=end_seq_time, height=0.5, color="red", edgecolor='black', label='zone 2')

    start_zone1 = enter_zone1(data,start_seq)
    start_zone1_time = start_zone1.iloc[:,0]
    plt.barh(y=0.25, width=np.array(start_seq_time) - np.array(start_zone1_time), left=start_zone1_time, height=0.5, color="green", edgecolor='black', label='zone 1')

    z1_df,z2_df = get_z1_z2(data)
    
    
    e1 = z1_df.iloc[1:]
    e2 = z2_df.iloc[:-1]


    start_pellet, end_pellet = pellet_interval(data,e1,e2)
    start_pellet, end_pellet = start_pellet.iloc[:,0], end_pellet.iloc[:,0]
    plt.barh(y=1.75, width=np.array(end_pellet) - np.array(start_pellet), left=start_pellet, height=0.5, color="purple", edgecolor='black', label='pellet')


    start_no_pellet, end_no_pellet = no_pellet_interval(data,e1,e2)
    start_no_pellet, end_no_pellet = start_no_pellet.iloc[:,0], end_no_pellet.iloc[:,0]
    plt.barh(y=1.75, width=np.array(end_no_pellet) - np.array(start_no_pellet), left=start_no_pellet, height=0.5, color="orange", edgecolor='black', label='no pellet')
   

    start_explo2, end_explo2 = explo_2(data,e1,e2)
    start_explo2_time, end_explo2_time = start_explo2.iloc[:,0], end_explo2.iloc[:,0]
    plt.barh(y=.25, width=np.array(end_explo2_time) - np.array(start_explo2_time), left=start_explo2_time, height=0.5, color="gray", edgecolor='black', label='explo 2')

    start_explo1, end_explo1 = explo_1(data)
    plt.barh(y=.25, width=np.array(end_explo1) - np.array(start_explo1), left=start_explo1, height=0.5, color="darkgray", edgecolor='black', label='explo 1')

    #plt.barh(y=0.15, width=np.array(z2_df) - np.array(z1_df), left=z1_df, height=0.5, color="yellow", edgecolor='black', label='z1 - z2')


    time_pellet = get_time_pellet(data,1)
    y_pellet = [1.75]*len(time_pellet)
    plt.scatter(time_pellet,y_pellet, edgecolor='black',color='purple')

    time_pellet = get_time_pellet(data,0)
    y_pellet = [1.75]*len(time_pellet)
    plt.scatter(time_pellet,y_pellet, edgecolor='black', color='orange')



    image_name = file_path.split('/')[-1]
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/images/time_sequences/FR5_I6/{image_name}.png' , dpi=300)



def save_matrix(file_path):

    data = load_data(file_path)

    all_actions_df = pd.DataFrame(columns=['time','type'])

    start_seq, end_seq = new_sequence(data)
    start_zone1 = enter_zone1(data,start_seq)
    z1_df,z2_df = get_z1_z2(data)
    e1 = z1_df.iloc[1:]
    e2 = z2_df.iloc[:-1]
    start_pellet, end_pellet = pellet_interval(data,e1,e2)
    start_no_pellet, end_no_pellet = no_pellet_interval(data,e1,e2)
    start_explo2, end_explo2 = explo_2(data,e1,e2)
    start_explo1, end_explo1 = explo_1(data)



    for __, row in start_seq.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['sequence']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in end_seq.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['zone 2']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_zone1.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['zone 1']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_pellet.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['pellet']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_no_pellet.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['no pellet']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_explo2.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['explo 2']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for current_time in start_explo1.values:
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['explo 1']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)


    #pd.set_option('display.max_rows', None)    # None means unlimited rows

    all_actions_df = all_actions_df.sort_values(by='time').reset_index(drop=True)
    all_actions_df2 = all_actions_df.iloc[1:,:]
    all_actions_df = all_actions_df.iloc[:-1,:]


    list = ['sequence', 'zone 2', 'pellet','no pellet', 'explo 2','explo 1','zone 1']
    matrix = np.zeros((7, 7))
    for x,y in zip(all_actions_df.iloc[:,1],all_actions_df2.iloc[:,1]):
        matrix[list.index(x)][list.index(y)] +=1


    # row_sums = np.sum(matrix, axis=1)
    # mask = row_sums != 0
    # matrix[mask] = matrix[mask] / row_sums[mask][:, np.newaxis]


    plt.figure(figsize=(10, 8))  # Set figure size
    sns.heatmap(matrix, annot=True, cmap='Greys', vmin=0, vmax=np.max(matrix), square=True,
                xticklabels=list, 
                yticklabels=list)

    plt.xlabel('STATE i+1')
    plt.ylabel('STATE i')

    plt.title('Transition matrix')
    plt.tight_layout()
    

    # image_name = file_path.split('/')[-1]
    # plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/images/transition_matrices/FR5_D1_3/{image_name}.png' , dpi=300)



def return_matrix(file_path):

    data = load_data(file_path)

    all_actions_df = pd.DataFrame(columns=['time','type'])

    start_seq, end_seq = new_sequence(data)
    start_zone1 = enter_zone1(data,start_seq)
    z1_df,z2_df = get_z1_z2(data)
    e1 = z1_df.iloc[1:]
    e2 = z2_df.iloc[:-1]
    start_pellet, end_pellet = pellet_interval(data,e1,e2)
    start_no_pellet, end_no_pellet = no_pellet_interval(data,e1,e2)
    start_explo2, end_explo2 = explo_2(data,e1,e2)
    start_explo1, end_explo1 = explo_1(data)



    for __, row in start_seq.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['sequence']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in end_seq.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['zone 2']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_zone1.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['zone 1']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_pellet.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['pellet']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_no_pellet.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['no pellet']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_explo2.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['explo 2']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for current_time in start_explo1.values:
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['explo 1']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)


    #pd.set_option('display.max_rows', None)    # None means unlimited rows

    all_actions_df = all_actions_df.sort_values(by='time').reset_index(drop=True)
    all_actions_df2 = all_actions_df.iloc[1:,:]
    all_actions_df = all_actions_df.iloc[:-1,:]


    list = ['sequence', 'zone 2', 'pellet','no pellet', 'explo 2','explo 1','zone 1']
    matrix = np.zeros((7, 7))
    for x,y in zip(all_actions_df.iloc[:,1],all_actions_df2.iloc[:,1]):
        matrix[list.index(x)][list.index(y)] +=1

    return matrix




def number_cycles(file_path):

    data = load_data(file_path)

    all_actions_df = pd.DataFrame(columns=['time','type'])

    start_seq, end_seq = new_sequence(data)
    start_zone1 = enter_zone1(data,start_seq)
    z1_df,z2_df = get_z1_z2(data)
    e1 = z1_df.iloc[1:]
    e2 = z2_df.iloc[:-1]
    start_pellet, end_pellet = pellet_interval(data,e1,e2)
    start_no_pellet, end_no_pellet = no_pellet_interval(data,e1,e2)
    start_explo2, end_explo2 = explo_2(data,e1,e2)
    start_explo1, end_explo1 = explo_1(data)



    for __, row in start_seq.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['sequence']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in end_seq.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['zone 2']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_zone1.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['zone 1']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_pellet.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['pellet']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_no_pellet.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['no pellet']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for __, row in start_explo2.iterrows():
        current_time = row['time']
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['explo 2']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)

    for current_time in start_explo1.values:
        new_row_df = pd.DataFrame({'time': [current_time], 'type': ['explo 1']})
        all_actions_df = pd.concat([all_actions_df, new_row_df], ignore_index=True)


    #pd.set_option('display.max_rows', None)    # None means unlimited rows

    all_actions_df = all_actions_df.sort_values(by='time').reset_index(drop=True)
    

    return all_actions_df