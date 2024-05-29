import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sys
import os 
from scipy.stats import percentileofscore

sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from behaviors_function import *




def calcium_sequence(start_seq, end_seq, trace_list):

    if len(start_seq) != 0 and len(end_seq) !=0:
        trace_val = []
        for start,end in zip (start_seq,end_seq):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)

        return np.mean(trace_val)
    else:
        return 0

def calcium_zone1(start_seq, start_zone1, trace_list):
    if len(start_zone1) != 0 and len(start_seq) !=0:
        trace_val = []
        for start,end in zip(start_zone1,start_seq):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)
        return np.mean(trace_val)
    else:
        return 0

def calcium_zone2(end_seq, end_zone2, trace_list):
    if len(end_seq) != 0 and len(end_zone2) !=0:
        trace_val = []
        for start,end in zip(end_seq, end_zone2):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)
        return np.mean(trace_val)
    else:
        return 0

def calcium_pellet(start_pellet, end_pellet, trace_list):
    if len(start_pellet) != 0 and len(end_pellet) !=0:
        trace_val = []
        for start,end in zip(start_pellet, end_pellet):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)
        return np.mean(trace_val)
    else:
        return 0

def calcium_no_pellet(start_no_pellet, end_no_pellet, trace_list):
    if len(start_no_pellet) != 0 and len(end_no_pellet) !=0:
        trace_val = []
        for start,end in zip(start_no_pellet, end_no_pellet):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)
        return np.mean(trace_val)
    else:
        return 0

def calcium_explo1(start_explo_1, end_explo_1, trace_list):
    if len(start_explo_1) != 0 and len(end_explo_1) !=0:
        trace_val = []
        for start,end in zip(start_explo_1, end_explo_1):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)
        return np.mean(trace_val)
    else:
        return 0

def calcium_explo2(start_explo_2, end_explo_2, trace_list):
    if len(start_explo_2) != 0 and len(end_explo_2) !=0:
        trace_val = []
        for start,end in zip(start_explo_2, end_explo_2):
            index_start = round(start/(1000*0.05))
            index_end = round(end/(1000*0.05))

            sub = trace_list[index_start:index_end+1]
            trace_val.extend(sub)
        return np.mean(trace_val)
    else:
        return 0






def apply_offset(lst, offset):
    if offset >= 0:
        index = round(offset/(0.05))
        end_index = len(lst)-1
        cut_index = end_index - index

        end_list = lst[:cut_index+1]
        begin_list = lst[cut_index+1:]

        offset_list = begin_list + end_list

    else : 
        cut_index = -round(offset/(0.05))
        begin_list = lst[cut_index+1:]
        end_list = lst[:cut_index+1]
        offset_list = begin_list + end_list

    return offset_list

def find_percentile(list, no_offset_val):
    percentile = percentileofscore(list, no_offset_val)
    return percentile

def list_percentiles(list):

    for offset in range(-1000, -200):
        offset /= 10
        #print(offset)
        offset_list = apply_offset(list, offset)
        val_seq = calcium_sequence(start_seq,end_seq,offset_list)
        val_zone1 = calcium_zone1(start_seq, start_zone1, offset_list)
        val_zone2 = calcium_zone2(end_seq, end_zone2, offset_list)
        val_pellet = calcium_pellet(start_pellet, end_pellet, offset_list)
        val_no_pellet = calcium_no_pellet(start_no_pellet, end_no_pellet, offset_list)
        val_explo1 = calcium_explo1(start_explo_1, end_explo_1, offset_list)
        val_explo2 = calcium_explo2(start_explo_2, end_explo_2, offset_list)


        offsets.append(offset)
        freq_sequence_list.append(val_seq)
        freq_zone1_list.append(val_zone1)
        freq_zone2_list.append(val_zone2)
        freq_pellet_list.append(val_pellet)
        freq_no_pellet_list.append(val_no_pellet)
        freq_explo1_list.append(val_explo1)
        freq_explo2_list.append(val_explo2)

    for offset in range(200, 1000):
        offset /= 10
        #print(offset)
        offset_list = apply_offset(list, offset)
        val_seq = calcium_sequence(start_seq,end_seq,offset_list)
        val_zone1 = calcium_zone1(start_seq, start_zone1, offset_list)
        val_zone2 = calcium_zone2(end_seq, end_zone2, offset_list)
        val_pellet = calcium_pellet(start_pellet, end_pellet, offset_list)
        val_no_pellet = calcium_no_pellet(start_no_pellet, end_no_pellet, offset_list)
        val_explo1 = calcium_explo1(start_explo_1, end_explo_1, offset_list)
        val_explo2 = calcium_explo2(start_explo_2, end_explo_2, offset_list)


        offsets.append(offset)
        freq_sequence_list.append(val_seq)
        freq_zone1_list.append(val_zone1)
        freq_zone2_list.append(val_zone2)
        freq_pellet_list.append(val_pellet)
        freq_no_pellet_list.append(val_no_pellet)
        freq_explo1_list.append(val_explo1)
        freq_explo2_list.append(val_explo2)


    percentile_sequence = find_percentile(freq_sequence_list, calcium_sequence(start_seq,end_seq,list))
    percentile_zone1 = find_percentile(freq_zone1_list, calcium_zone1(start_seq, start_zone1, list))
    percentile_zone2 = find_percentile(freq_zone2_list, calcium_zone2(end_seq, end_zone2, list))
    percentile_pellet = find_percentile(freq_pellet_list, calcium_pellet(start_pellet, end_pellet, list))
    percentile_no_pellet = find_percentile(freq_no_pellet_list, calcium_no_pellet(start_no_pellet, end_no_pellet, list))
    percentile_explo1 = find_percentile(freq_explo1_list, calcium_explo1(start_explo_1, end_explo_1, list))
    percentile_explo2 = find_percentile(freq_explo2_list, calcium_explo2(start_explo_2, end_explo_2, list))

    output = [percentile_sequence, percentile_zone1, percentile_zone2, percentile_pellet, percentile_no_pellet, percentile_explo1, percentile_explo2]

    return output

def save_percentiles(output_file):
    rows = []

    for i, list in trace_df.iterrows():
        list = list.tolist()

        row = list_percentiles(list)
        print(i, row)

        rows.append(row)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False, header=False)




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

name = 'L0'

csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
df = pd.read_csv(csv_file_path)

row_numbers = extract_sessions(name)
row_numbers = row_numbers[5:]
print(row_numbers)

for session in row_numbers:
    print('###########'*3) 
    print(session)

    trace_path = df.iloc[session,1]
    time_path = df.iloc[session,2]
    dat_path = df.iloc[session,0]


    trace_df = pd.read_csv(trace_path, header=None)
    time_df = pd.read_csv(time_path, header=None)


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

    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/trace/percentiles/{name}/S{session+1}.csv'
    save_percentiles(output_file)







