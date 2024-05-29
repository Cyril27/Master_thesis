import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import csv
import sys 
from scipy.stats import percentileofscore


sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from behaviors_function import *


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


def time_val(list):
    t_list = []
    val_list = []
    for id, elem in enumerate(list):
        if elem != 0:

            t_list.append(id*0.05)
            val_list.append(elem)

    return t_list, val_list


def apply_offset(lst, offset, list2):
    max_val = max(lst)
    end_list = [x + offset for x in lst]
    begin_list = []

    for element in end_list[:]:  # Iterate over a copy of end_list to avoid modifying it
        if element > max_val:
            end_list.remove(element)
            begin_list.append(element - max_val)

    end_list2 = list2[:len(end_list)]
    begin_list2 = list2[len(end_list):]

    return begin_list + end_list, begin_list2 + end_list2

def find_percentile(list, no_offset_val):
    percentile = percentileofscore(list, no_offset_val)
    return percentile

def list_percentiles(data,list, val_list):

    offsets = []
    freq_sequence_list = []
    freq_moving_list = []
    freq_dispenser_list = []
    freq_idle_list = []

    freq_zone1_list = []
    freq_zone2_list = []
    freq_pellet_list = []
    freq_no_pellet_list = []
    freq_explo1_list = []
    freq_explo2_list = []

    for offset in range(-500, -200):
        offset /= 10
        #print(offset)
        offset_list, offset_val_list = apply_offset(list, offset, val_list)

        freq_seq= freq_sequence_decon(data, offset_list,start_seq, end_seq, time_seq, offset_val_list)
        freq_z1 = freq_zone1_decon(data, offset_list, start_seq, start_zone1, time_z1, offset_val_list)
        freq_z2 = freq_zone2_decon(data, offset_list, end_seq, end_zone2, time_z2, offset_val_list)
        freq_p = freq_pellet_decon(data, offset_list, start_pellet, end_pellet, time_p, offset_val_list)
        freq_nop = freq_no_pellet_decon(data, offset_list, start_no_pellet, end_no_pellet, time_nop, offset_val_list)
        freq_exp1 = freq_explo1_decon(data, offset_list, start_explo_1, end_explo_1, time_exp1, offset_val_list)
        freq_exp2 = freq_explo2_decon(data, offset_list, start_explo_2, end_explo_2, time_exp2, offset_val_list)


        offsets.append(offset)
        freq_sequence_list.append(freq_seq)
        freq_zone1_list.append(freq_z1)
        freq_zone2_list.append(freq_z2)
        freq_pellet_list.append(freq_p)
        freq_no_pellet_list.append(freq_nop)
        freq_explo1_list.append(freq_exp1)
        freq_explo2_list.append(freq_exp2)

    for offset in range(200, 500):
        offset /= 10
        #print(offset)
        offset_list, offset_val_list = apply_offset(list, offset, val_list)
        freq_seq = freq_sequence_decon(data, offset_list,start_seq, end_seq, time_seq, offset_val_list)
        freq_z1 = freq_zone1_decon(data, offset_list, start_seq, start_zone1, time_z1, offset_val_list)
        freq_z2 = freq_zone2_decon(data, offset_list, end_seq, end_zone2, time_z2, offset_val_list)
        freq_p = freq_pellet_decon(data, offset_list, start_pellet, end_pellet, time_p, offset_val_list)
        freq_nop = freq_no_pellet_decon(data, offset_list, start_no_pellet, end_no_pellet, time_nop, offset_val_list)
        freq_exp1 = freq_explo1_decon(data, offset_list, start_explo_1, end_explo_1, time_exp1, offset_val_list)
        freq_exp2 = freq_explo2_decon(data, offset_list, start_explo_2, end_explo_2, time_exp2, offset_val_list)


        offsets.append(offset)
        freq_sequence_list.append(freq_seq)
        freq_zone1_list.append(freq_z1)
        freq_zone2_list.append(freq_z2)
        freq_pellet_list.append(freq_p)
        freq_no_pellet_list.append(freq_nop)
        freq_explo1_list.append(freq_exp1)
        freq_explo2_list.append(freq_exp2)


    percentile_sequence = find_percentile(freq_sequence_list, freq_sequence_decon(data, list, start_seq, end_seq, time_seq, val_list))
    percentile_zone1 = find_percentile(freq_zone1_list, freq_zone1_decon(data, list, start_seq, start_zone1, time_z1, val_list))
    percentile_zone2 = find_percentile(freq_zone2_list, freq_zone2_decon(data, list, end_seq, end_zone2, time_z2, val_list))
    percentile_pellet = find_percentile(freq_pellet_list, freq_pellet_decon(data, list, start_pellet, end_pellet, time_p, val_list))
    percentile_no_pellet = find_percentile(freq_no_pellet_list, freq_no_pellet_decon(data, list, start_no_pellet, end_no_pellet, time_nop, val_list))
    percentile_explo1 = find_percentile(freq_explo1_list, freq_explo1_decon(data, list, start_explo_1, end_explo_1, time_exp1, val_list))
    percentile_explo2 = find_percentile(freq_explo2_list, freq_explo2_decon(data, list, start_explo_2, end_explo_2, time_exp2, val_list))

    output = [percentile_sequence, percentile_zone1, percentile_zone2, percentile_pellet, percentile_no_pellet, percentile_explo1, percentile_explo2]

    return output

def save_percentiles(output_file):
    rows = []
    

    for (i,t_list), (j,val_list) in zip(time_decon_df.iterrows(), val_decon_df.iterrows()):

        t_list = t_list.dropna()
        t_list = t_list.tolist()

        val_list = val_list.dropna()
        val_list = val_list.tolist() 
        
        row = list_percentiles(data,t_list, val_list)

        print(i, row)
        rows.append(row)

    df_percentile = pd.DataFrame(rows)
    df_percentile.to_csv(output_file, index=False, header=False)



def list_freq_4(data,list, val_list):
    freq_seq = freq_sequence_decon(data, list, start_seq, end_seq, time_seq,val_list)
    freq_mv = freq_moving_decon(list, start_seq, end_seq, start_zone1, end_zone2, time_mv, val_list)
    freq_food = freq_dispenser_decon(data,list, start_pellet, end_pellet, start_no_pellet, end_no_pellet, time_food , val_list)
    freq_explo = freq_idle_decon(data,list, start_explo_1, end_explo_1, start_explo_2, end_explo_2, time_explo, val_list)
   

    return [freq_seq, freq_mv, freq_food, freq_explo]


def save_freq_4(output_file):
    rows = []
    

    for (i,t_list), (j,val_list) in zip(time_decon_df.iterrows(), val_decon_df.iterrows()):

        t_list = t_list.dropna()
        t_list = t_list.tolist()

        val_list = val_list.dropna()
        val_list = val_list.tolist() 
        
        row = list_freq_4(data,t_list, val_list)

        print(i, row)
        rows.append(row)

    df_percentile = pd.DataFrame(rows)
    df_percentile.to_csv(output_file, index=False, header=False)




name = 'L3'

csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
df = pd.read_csv(csv_file_path)

row_numbers = extract_sessions(name)
print(row_numbers)

for session in row_numbers: 
    print('###########'*3) 
    print(session)

    dat_path = df.iloc[session ,0]
    csv_decon_spike = df.iloc[session,6]
    time_path = df.iloc[session,2]
    csv_spike = df.iloc[session,3]


    time_df = pd.read_csv(time_path, header=None)
    decon_df = pd.read_csv(csv_decon_spike, header=None)

    time_decon_list = []
    val_decon_list = []
    for i,row in decon_df.iterrows():

        t_list,val_list = time_val(row)
        time_decon_list.append(t_list)
        val_decon_list.append(val_list)

    time_decon_df = pd.DataFrame(time_decon_list)
    val_decon_df = pd.DataFrame(val_decon_list)


    # time_path = df.iloc[session-1 ,2]
    # df_time = pd.read_csv(time_path)
    # end_time = df_time.shape[1]*0.05
    # time_series = np.arange(0,end_time, 0.05)



    data = load_data(dat_path)

    start_seq, end_seq = new_sequence(data)
    start_zone1 = enter_zone1(data,start_seq)
    end_zone2 = enter_zone2(data,end_seq)

    start_seq, end_seq = start_seq.iloc[:,0], end_seq.iloc[:,0]
    start_zone1 = start_zone1.iloc[:,0]
    end_zone2 = end_zone2.iloc[:,0]

    z1_df,z2_df = get_z1_z2(data)
    e1 = z1_df.iloc[1:]
    e2 = z2_df.iloc[:-1]

    start_pellet, end_pellet = pellet_interval(data,e1,e2)
    start_pellet, end_pellet = start_pellet.iloc[:,0], end_pellet.iloc[:,0]
    start_no_pellet, end_no_pellet = no_pellet_interval(data,e1,e2)
    start_no_pellet, end_no_pellet = start_no_pellet.iloc[:,0], end_no_pellet.iloc[:,0]

    start_explo_1, end_explo_1 = explo_1(data)
    start_explo_2, end_explo_2 = explo_2(data,e1,e2)
    start_explo_2, end_explo_2 = start_explo_2.iloc[:,0], end_explo_2.iloc[:,0]

    time_seq = time_sequence(data)
    time_z1 = time_zone1(data)
    time_z2 = time_zone2(data)
    time_p = time_pellet(data)
    time_nop = time_nopellet(data)
    time_exp1 = time_explo1(data)
    time_exp2 = time_explo2(data)

    time_food = time_dispenser(data)
    time_mv = time_moving(data)
    time_explo = time_idle(data)



    #### for percentiles
    # output_file = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/percentiles/{name}/S{session+1}.csv'
    # save_percentiles(output_file)


    ### for frequencies_4
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/frequencies_4/{name}/freq_S{session+1}.csv'
    save_freq_4(output_file)




