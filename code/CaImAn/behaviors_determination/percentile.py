import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
import sys
from collections import Counter
import csv

sys.path.append('/Users/cyrilvanleer/Desktop/Thesis/dat_files/code/')
from behaviors_function import *


name = 'I2'

csv_file_path = f'/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_neurons/{name}.csv'  
df = pd.read_csv(csv_file_path)






session = 12     
row_val = 0

dat_path = df.iloc[session-1 ,0]
csv_spike = df.iloc[session-1 ,3]

with open(csv_spike, mode='r') as csvfile:
    reader = csv.reader(csvfile)
    row_count = sum(1 for row in reader)



def get_spike(path, row):
    df = pd.read_csv(path, skiprows=lambda x: x != row, header=None, nrows=1)
    list_from_df = df.values.tolist()[0] 
    return list_from_df

def apply_offset(lst, offset):
    max_val = max(lst)
    end_list = [x + offset for x in lst]
    begin_list = []

    for element in end_list[:]:  # Iterate over a copy of end_list to avoid modifying it
        if element > max_val:
            end_list.remove(element)
            begin_list.append(element - max_val)

    return begin_list + end_list


#list = get_spike(csv_spike,row_val)
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



time_mv = time_moving(data)
time_disp = time_dispenser(data)
time_id = time_idle(data)



def mean_time(start_zone1):
    time = 0
    occ = len(start_zone1)

    for i in range(1,occ):
        add = start_zone1[i] - start_zone1[i-1]
        time += add
    return time/occ        

def mean_time2(start_zone1):

    list = []
    occ = len(start_zone1)

    for i in range(1,occ):
        add = start_zone1[i] - start_zone1[i-1]
        list.append(add)

    lower_bound = np.percentile(list, 5)
    upper_bound = np.percentile(list, 95)

    cleaned_list = [x for x in list if lower_bound <= x <= upper_bound]

    return np.mean(cleaned_list)

#print(mean_time(start_zone1))
#print(mean_time2(start_zone1))

def find_percentile(list, no_offset_val):
    percentile = percentileofscore(list, no_offset_val)
    return percentile

def list_percentiles(data,list):

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

    for offset in range(-1000, -200):
        offset /= 10
        #print(offset)
        offset_list = apply_offset(list, offset)
        freq_seq = freq_sequence(data, offset_list,start_seq, end_seq, time_seq)
        freq_z1 = freq_zone1(data, offset_list, start_seq, start_zone1, time_z1)
        freq_z2 = freq_zone2(data, offset_list, end_seq, end_zone2, time_z2)
        freq_p = freq_pellet(data, offset_list, start_pellet, end_pellet, time_p)
        freq_nop = freq_no_pellet(data, offset_list, start_no_pellet, end_no_pellet, time_nop)
        freq_exp1 = freq_explo1(data, offset_list, start_explo_1, end_explo_1, time_exp1)
        freq_exp2 = freq_explo2(data, offset_list, start_explo_2, end_explo_2, time_exp2)


        offsets.append(offset)
        freq_sequence_list.append(freq_seq)
        freq_zone1_list.append(freq_z1)
        freq_zone2_list.append(freq_z2)
        freq_pellet_list.append(freq_p)
        freq_no_pellet_list.append(freq_nop)
        freq_explo1_list.append(freq_exp1)
        freq_explo2_list.append(freq_exp2)

    for offset in range(200, 1000):
        offset /= 10
        #print(offset)
        offset_list = apply_offset(list, offset)
        freq_seq = freq_sequence(data, offset_list,start_seq, end_seq, time_seq)
        freq_z1 = freq_zone1(data, offset_list, start_seq, start_zone1, time_z1)
        freq_z2 = freq_zone2(data, offset_list, end_seq, end_zone2, time_z2)
        freq_p = freq_pellet(data, offset_list, start_pellet, end_pellet, time_p)
        freq_nop = freq_no_pellet(data, offset_list, start_no_pellet, end_no_pellet, time_nop)
        freq_exp1 = freq_explo1(data, offset_list, start_explo_1, end_explo_1, time_exp1)
        freq_exp2 = freq_explo2(data, offset_list, start_explo_2, end_explo_2, time_exp2)


        offsets.append(offset)
        freq_sequence_list.append(freq_seq)
        freq_zone1_list.append(freq_z1)
        freq_zone2_list.append(freq_z2)
        freq_pellet_list.append(freq_p)
        freq_no_pellet_list.append(freq_nop)
        freq_explo1_list.append(freq_exp1)
        freq_explo2_list.append(freq_exp2)

    # file_path = "/Users/cyrilvanleer/Desktop/Thesis/neurons/test.csv"

    # # Write the list to the CSV file
    # with open(file_path, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(freq_no_pellet_list)




    percentile_sequence = find_percentile(freq_sequence_list, freq_sequence(data, list, start_seq, end_seq, time_seq))
    percentile_zone1 = find_percentile(freq_zone1_list, freq_zone1(data, list, start_seq, start_zone1, time_z1))
    percentile_zone2 = find_percentile(freq_zone2_list, freq_zone2(data, list, end_seq, end_zone2, time_z2))
    percentile_pellet = find_percentile(freq_pellet_list, freq_pellet(data, list, start_pellet, end_pellet, time_p))
    percentile_no_pellet = find_percentile(freq_no_pellet_list, freq_no_pellet(data, list, start_no_pellet, end_no_pellet, time_nop))
    percentile_explo1 = find_percentile(freq_explo1_list, freq_explo1(data, list, start_explo_1, end_explo_1, time_exp1))
    percentile_explo2 = find_percentile(freq_explo2_list, freq_explo2(data, list, start_explo_2, end_explo_2, time_exp2))





    output = [percentile_sequence, percentile_zone1, percentile_zone2, percentile_pellet, percentile_no_pellet, percentile_explo1, percentile_explo2]




    return output

def save_percentiles():
    rows = []
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/{name}/S{session}.csv'

    for i in range(row_count):
        #print(i)
        list = get_spike(csv_spike,i)
        row = list_percentiles(data,list)

        print(i, row)
        rows.append(row)

    df_percentile = pd.DataFrame(rows)

    #df_percentile.to_csv(output_file, index=False, header=False)


#save_percentiles()



def list_freq(data,list):
    freq_seq = freq_sequence(data, list,start_seq, end_seq, time_seq)
    freq_z1 = freq_zone1(data, list, start_seq, start_zone1, time_z1)
    freq_z2 = freq_zone2(data, list, end_seq, end_zone2, time_z2)
    freq_p = freq_pellet(data, list, start_pellet, end_pellet, time_p)
    freq_nop = freq_no_pellet(data, list, start_no_pellet, end_no_pellet, time_nop)
    freq_exp1 = freq_explo1(data, list, start_explo_1, end_explo_1, time_exp1)
    freq_exp2 = freq_explo2(data, list, start_explo_2, end_explo_2, time_exp2)

    return [freq_seq, freq_z1, freq_z2, freq_p, freq_nop, freq_exp1, freq_exp2]

def save_freq():
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/frequencies/{name}/freq_S{session}.csv'

    freq_list = []
    for i in range(row_count):
        list = get_spike(csv_spike,i)
        freq_row = list_freq(data,list)
        freq_list.append(freq_row)
        print(i, freq_row)


    df_percentile = pd.DataFrame(freq_list)

    df_percentile.to_csv(output_file, index=False, header=False)



#save_freq()



def list_freq_4(data,list):
    freq_seq = freq_sequence(data, list,start_seq, end_seq, time_seq)
    freq_mv = freq_moving(list, start_seq, end_seq, start_zone1, end_zone2, time_mv)
    freq_food = freq_dispenser(data,list, start_pellet, end_pellet, start_no_pellet, end_no_pellet, time_disp)
    freq_explo = freq_idle(data,list, start_explo_1, end_explo_1, start_explo_2, end_explo_2, time_id)
   

    return [freq_seq, freq_mv, freq_food, freq_explo]



def save_freq_4():
    output_file = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/frequencies_4/{name}/freq_S{session}.csv'

    freq_list = []
    for i in range(row_count):
        list = get_spike(csv_spike,i)
        freq_row = list_freq_4(data,list)
        freq_list.append(freq_row)
        print(i, freq_row)


    df_percentile = pd.DataFrame(freq_list)

    df_percentile.to_csv(output_file, index=False, header=False)


save_freq_4()