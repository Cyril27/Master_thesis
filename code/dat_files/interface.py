import streamlit as st 
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import mplcursors

column_names = ['family', 'num', 'P', 'V', 'L', 'R', 'T','W', 'X', 'Y', 'Z']
family_names = ['','Light (1)', 'Lever (2)','', 'Pellet dispenser (4)','', 'Divers CA (6)','','', 'Zone (9)', 'MISC (10)', 'Divers non stockes (11)','','', 'Rearing (14)']


st.title("Manipulate .dat files")

file_path = st.text_input('Path of the file to analyse','/Volumes/CalciumImaging_Sandra_Cold/EXP2 Fixed-Ratio results/2020_7_2020_10_A2A/Imetronic .dat Jul-Ag2020/T3FR5-D10-TTL-DBL-L0-06-08-20_01.dat')

def load_data(file_path):
    data = pd.read_csv(file_path, delimiter='\t', skiprows=12, header=None, names=column_names)
    return data


data = load_data(file_path)


if st.toggle("show data"):
    st.dataframe(data, width=1300)
    

st.header(".dat file subset")


family = st.selectbox("Select a family",('Light (1)', 'Lever (2)', 'Pellet dispenser (4)', 'Divers CA (6)', 'Zone (9)', 'MISC (10)', 'Rearing (14)', 'Divers non stockes (11)'))




def filter_data(data,fam,param_list):

    index = family_names.index(fam) 
    subset = data[data.iloc[:, 0] == index]

    for i in range(len(param_list)):
        if param_list[i] >-1:
            subset = subset[subset.iloc[:, i+1] == param_list[i]]

    return subset






if family == 'Light (1)':

    st.header("Parameters for Light (1)")
    param = 10*[-1]

    col1,col2 = st.columns(2)

    with col1:

        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use P as parameter'):
            P = st.number_input("P", step=1, min_value=0,max_value=1)
            param[1] = P

    with col2:  
        if st.toggle('Use L as parameter'):
            L = st.number_input("L", step=1, min_value=0)
            param[3] = L
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0)
            param[5] = T  




if family == 'Lever (2)':

    st.header("Parameters for Lever (2)")
    param = 10*[-1]
    
    col1,col2 = st.columns(2)

    with col1:
        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use P as parameter'):
            P = st.number_input("", step=1, min_value=0,max_value=1)
            param[1] = P
        if st.toggle('Use V as parameter'):
            V = st.number_input("V", step=1, min_value=0,max_value=1)
            param[2] = V
        if st.toggle('Use L as parameter'):
            L = st.number_input("L", step=1, min_value=0)
            param[3] = L
         

    with col2:
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0)
            param[5] = T 
        if st.toggle('Use W as parameter'):
            W = st.number_input("W", step=1, min_value=0)   
            param[6] = W  
        if st.toggle('Use X as parameter'):
            X = st.number_input("X", step=1, min_value=0)
            param[7] = X
        if st.toggle('Use Y as parameter'):
            Y = st.number_input("Y", step=1, min_value=0)
            param[8] = Y
        

if family == 'Pellet dispenser (4)':

    st.header("Parameters for pellet dispenser (4)")
    param = 10*[-1]
    
    col1,col2 = st.columns(2)

    with col1:
        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use P as parameter'):
            P = st.number_input("", step=1, min_value=0,max_value=1)
            param[1] = P
        if st.toggle('Use V as parameter'):
            V = st.number_input("V", step=1, min_value=0,max_value=1)
            param[2] = V
        if st.toggle('Use L as parameter'):
            L = st.number_input("L", step=1, min_value=0)
            param[3] = L
        if st.toggle('Use R as parameter'):
            R = st.number_input("R", step=1, min_value=0)
            param[4] = R

    with col2:
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0)
            param[5] = T 
        if st.toggle('Use W as parameter'):
            W = st.number_input("W", step=1, min_value=0)   
            param[6] = W    
        if st.toggle('Use X as parameter'):
            X = st.number_input("X", step=1, min_value=0)
            param[7] = X   
        if st.toggle('Use Y as parameter'):
            Y = st.number_input("Y", step=1, min_value=0)
            param[8] = Y
        if st.toggle('Use Z as parameter'):
            Z = st.number_input("Z", step=1, min_value=0)
            param[9] = Z



if family == 'Divers CA (6)':

    st.header("Parameters for  Divers CA (6)")
    param = 10*[-1]
    
    col1,col2 = st.columns(2)

    with col1:

        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use P as parameter'):
            P = st.number_input("P", step=1, min_value=0,max_value=1)
            param[1] = P

    with col2:  
        if st.toggle('Use L as parameter'):
            L = st.number_input("L", step=1, min_value=0)
            param[3] = L
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0) 
            param[5] = T 


if family == 'Zone (9)':

    st.header("Parameters for  Zone (9)")
    param = 10*[-1]
    
    col1,col2 = st.columns(2)

    with col1:
        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use P as parameter'):
            P = st.number_input("", step=1, min_value=0,max_value=1)
            param[1] = P
        if st.toggle('Use V as parameter'):
            V = st.number_input("V", step=1, min_value=0,max_value=2)
            param[2] = V
        if st.toggle('Use L as parameter'):
            L = st.number_input("L", step=1, min_value=0)
            param[3] = L
        if st.toggle('Use R as parameter'):
            R = st.number_input("R", step=1, min_value=0)
            param[4] = R

    with col2:
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0)
            param[5] = T 
        if st.toggle('Use W as parameter'):
            W = st.number_input("W", step=1, min_value=0)   
            param[6] = W    
        if st.toggle('Use X as parameter'):
            X = st.number_input("X", step=1, min_value=0)
            param[7] = X   
        if st.toggle('Use Y as parameter'):
            Y = st.number_input("Y", step=1, min_value=0)
            param[8] = Y
        if st.toggle('Use Z as parameter'):
            Z = st.number_input("Z", step=1, min_value=0)
            param[9] = Z



if family == 'MISC (10)':

    st.header("Parameters for MISC (10)")
    param = 10*[-1]
    
    col1,col2 = st.columns(2)

    with col1:
        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0) 
            param[5] = T    
        if st.toggle('Use X as parameter'):
            X = st.number_input("X", step=1, min_value=0)
            param[7] = X 

    with col2:
        if st.toggle('Use Y as parameter'):
            Y = st.number_input("Y", step=1, min_value=0)
            param[8] = Y 
        if st.toggle('Use Z as parameter'):
            Z = st.number_input("Z", step=1, min_value=0)
            param[9] = Z 


if family == 'Divers non stockes (11)':

    st.header("Parameters for  Divers non stockes (11)")
    param = 10*[-1]
    
    col1,col2 = st.columns(2)

    with col1:

        if st.toggle('Use num as parameter'):
            num = st.number_input("num", step=1, min_value=0)
            param[0] = num
        if st.toggle('Use P as parameter'):
            P = st.number_input("P", step=1, min_value=0,max_value=1)
            param[1] = P

    with col2:  
        if st.toggle('Use L as parameter'):
            L = st.number_input("L", step=1, min_value=0)
            param[3] = L
        if st.toggle('Use T as parameter'):
            T = st.number_input("T", step=1, min_value=0) 
            param[5] = T 


filtered_data = filter_data(data,family,param)
if st.toggle('Show filtered data'):
    st.dataframe(filtered_data)

col = st.selectbox("Column to use as Y axis",column_names)



fig, ax = plt.subplots()
ax.scatter(filtered_data.index, filtered_data.iloc[:, column_names.index(col)])
ax.set_xlabel('Time (ms)')
ax.set_ylabel(col)
ax.set_title(f'Scatter Plot of Row Index vs. {col}')

st.write("Use the sliders below to zoom in on the plot:")
xlim = st.slider("X-Axis Range", min_value=min(filtered_data.index), max_value=max(filtered_data.index), value=(min(filtered_data.index), max(filtered_data.index)))
ylim = st.slider(f"{col} Range", min_value=min(filtered_data[col]), max_value=max(filtered_data[col]), value=(min(filtered_data[col]), max(filtered_data[col])))
ax.set_xlim(xlim)
ax.set_ylim(ylim)

st.pyplot(fig)




st.header("Multiple families in the subplot")


index_list = [2, 4, 9]

family_list = ['Lever', 'Dispenser', 'Zone']

options = st.multiselect(
    'Events types to show',
    family_list)


selected_indices = [index_list[family_list.index(option)] for option in options]

subset = data[data.iloc[:,0].isin(selected_indices)]
st.dataframe(subset)