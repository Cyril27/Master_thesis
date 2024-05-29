import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


name = 'I2'

folder = 'trace'


def plot_compare():

    session = 12

    percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/{name}/S{session}.csv'
    percentile_df = pd.read_csv(percentile_path, header=None)


    #decon_percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/percentiles/{name}/S{session}.csv'
    decon_percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session}.csv'
    decon_percentile_df = pd.read_csv(decon_percentile_path, header=None)


    def replace_values(val):
        if val > 95:
            return 1
        elif val < 5:
            return -1
        else:
            return 0

    bin_percentile_df = percentile_df.applymap(replace_values)
    bin_decon_percentile_df = decon_percentile_df.applymap(replace_values)

    plt.figure()
    sns.heatmap(bin_percentile_df, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title('Percentile')


    plt.figure()
    sns.heatmap(bin_decon_percentile_df, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title('Decon percentile')

    plt.show()


    from sklearn.metrics.pairwise import cosine_similarity

    # Assuming df1 and df2 are your DataFrames
    # Convert DataFrames to numpy arrays
    matrix1 = percentile_df.to_numpy()
    matrix2 = decon_percentile_df.to_numpy()

    similarity_matrix = cosine_similarity(matrix1, matrix2)

    # mean_similarity = similarity_matrix.mean()
    # print("Mean Cosine Similarity:", mean_similarity)

    # correlation_matrix = percentile_df.corrwith(decon_percentile_df, axis=1)
    # mean_correlation = correlation_matrix.mean()
    # print("Mean Pearson Correlation Coefficient:", mean_correlation)

    correlation_coefficient = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
    print("Correlation coefficient between the two matrices:", correlation_coefficient)

    plt.figure(figsize=(8, 6))
    plt.scatter(percentile_df.values.flatten(), decon_percentile_df.values.flatten(), color='blue', alpha=0.5)
    plt.xlabel('Percentiles from spikes')
    plt.ylabel('Percentiles from deconvoluted spikes')

    plt.show()

#plot_compare()


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

def global_and_behavior_corr():


    row_numbers = extract_sessions(name)

    result_list = []
    global_corr_list = []

    for session in row_numbers:

        percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/{name}/S{session+1}.csv'
        percentile_df = pd.read_csv(percentile_path, header=None)

        decon_percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session+1}.csv'
        decon_percentile_df = pd.read_csv(decon_percentile_path, header=None)

        matrix1 = percentile_df.to_numpy()
        matrix2 = decon_percentile_df.to_numpy()

        num_rows, num_cols = matrix1.shape

        correlation_coefficients = []
        for i in range(num_cols):
            correlation_coefficient = np.corrcoef(matrix1[:, i], matrix2[:, i])[0, 1]
            correlation_coefficients.append(correlation_coefficient)

        result_list.append(correlation_coefficients)

        global_corr = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
        global_corr_list.append(global_corr)

    result_df = pd.DataFrame(result_list)
    global_corr_df = pd.DataFrame(global_corr_list)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.heatmap(result_df, cmap='coolwarm', annot=True, fmt=".2f", ax=axs[0])
    axs[0].set_title('Behavior correlation')
    axs[0].set_ylabel('Session')
    axs[0].set_xticks(np.arange(len(result_df.columns)) + 0.5)
    axs[0].set_xticklabels(['sequence','zone 1', 'zone 2', 'pellet', 'no pellet', 'explo1', 'explo2'], rotation='vertical')
    axs[0].set_yticks(np.arange(len(result_df)) + 0.5)
    axs[0].set_yticklabels(['1','2', '3', '4', '5', '6', '7', '8'])

    sns.heatmap(global_corr_df, cmap='coolwarm', annot=True, fmt=".2f", ax=axs[1])
    axs[1].set_title('Global Correlation')
    axs[1].set_yticks(np.arange(len(result_df)) + 0.5)
    axs[1].set_yticklabels(['1','2', '3', '4', '5', '6', '7', '8'])

    plt.tight_layout()
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/compare/percentiles/{name}.png' , dpi=300)
    plt.show()


#global_and_behavior_corr()

def average_global():
    final_matrix = np.zeros((8,4))
    final_global = []


    name_list = ['I2','I3','I6']
    #name_list = ['L0','L2','L3']

    for name in name_list:
        row_numbers = extract_sessions(name)

        result_list = []
        global_corr_list = []

        for session in row_numbers:


            percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/{name}/S{session+1}.csv'
            percentile_df = pd.read_csv(percentile_path, header=None)

            decon_percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session+1}.csv'
            decon_percentile_df = pd.read_csv(decon_percentile_path, header=None)

            matrix1 = percentile_df.to_numpy()
            matrix2 = decon_percentile_df.to_numpy()

            #np.random.shuffle(matrix2)

            


            ### seq
            corr_seq = np.corrcoef(matrix1[:, 0], matrix2[:, 0])[0, 1]


            ### zones
            concatenated_col1 = np.concatenate((matrix1[:, 1], matrix1[:, 2]))
            concatenated_col2 = np.concatenate((matrix2[:, 1], matrix2[:, 2]))

            corr_zones = np.corrcoef(concatenated_col1, concatenated_col2)[0, 1]


            ### food
            concatenated_col1 = np.concatenate((matrix1[:, 3], matrix1[:, 4]))
            concatenated_col2 = np.concatenate((matrix2[:, 3], matrix2[:, 4]))

            corr_food = np.corrcoef(concatenated_col1, concatenated_col2)[0, 1]


            ### explo
            concatenated_col1 = np.concatenate((matrix1[:, 5], matrix1[:, 6]))
            concatenated_col2 = np.concatenate((matrix2[:, 5], matrix2[:, 6]))

            corr_explo = np.corrcoef(concatenated_col1, concatenated_col2)[0, 1]

            result_list.append([corr_seq, corr_zones, corr_food, corr_explo])



            global_corr = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
            global_corr_list.append(global_corr)



        final_matrix += result_list
        final_global.append(global_corr_list)


    final_matrix /= len(name_list)

    plt.show()

    result_df = pd.DataFrame(final_matrix)
    global_corr_df = pd.DataFrame(final_global)
    final_global = pd.DataFrame(global_corr_df.mean(axis=0))

    fig, axs = plt.subplots(1, 2, figsize=(6, 6))

    sns.heatmap(result_df, cmap='Reds', annot=True, fmt=".2f", ax=axs[0])
    #axs[0].set_title('Behavior correlation')
    #axs[0].set_ylabel('Session')
    axs[0].set_xticks(np.arange(len(result_df.columns)) + 0.5)
    axs[0].set_xticklabels(['Sequence','Zones', 'Food', 'Explo'], rotation=45)
    # axs[0].set_yticks(np.arange(len(result_df)) + 0.5)
    # axs[0].set_yticklabels(['1','2', '3', '4', '5', '6', '7', '8'])
    axs[0].set_yticks([])

    sns.heatmap(final_global, cmap='Reds', annot=True, fmt=".2f", ax=axs[1])
    #axs[1].set_title('Global Correlation')
    #axs[1].set_yticks(np.arange(len(result_df)) + 0.5)
    #axs[1].set_yticklabels(['1','2', '3', '4', '5', '6', '7', '8'])
    axs[1].set_yticks([])
    axs[1].set_xticks([])

    plt.tight_layout()
    #plt.savefig(f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/images/compare/percentiles/{name}.png' , dpi=300)
    plt.show()



name_list = ['I2','I3','I6']
#name_list = ['L0','L2','L3']

total_list = []

for name in name_list:

    row_numbers = extract_sessions(name)

    
    global_corr_list = []
    result_list = []
    for session in row_numbers:

        

        percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/percentiles/{name}/S{session+1}.csv'
        percentile_df = pd.read_csv(percentile_path, header=None)

        decon_percentile_path = f'/Users/cyrilvanleer/Desktop/Thesis/{folder}/percentiles/{name}/S{session+1}.csv'
        decon_percentile_df = pd.read_csv(decon_percentile_path, header=None)

        matrix1 = percentile_df.to_numpy()
        matrix2 = decon_percentile_df.to_numpy()

        matrix2_shuffled = np.copy(matrix2)
        np.random.shuffle(matrix2_shuffled)


        ### seq
        corr_seq = np.corrcoef(matrix1[:, 0], matrix2[:, 0])[0, 1]
        corr_seq_shuffled = np.corrcoef(matrix1[:, 0], matrix2_shuffled[:, 0])[0, 1]


        ### zones
        concatenated_col1 = np.concatenate((matrix1[:, 1], matrix1[:, 2]))
        concatenated_col2 = np.concatenate((matrix2[:, 1], matrix2[:, 2]))
        concatenated_col2_shuffled = np.concatenate((matrix2_shuffled[:, 1], matrix2_shuffled[:, 2]))

        corr_zones = np.corrcoef(concatenated_col1, concatenated_col2)[0, 1]
        corr_zones_shuffled = np.corrcoef(concatenated_col1, concatenated_col2_shuffled)[0, 1]

        ### food
        concatenated_col1 = np.concatenate((matrix1[:, 3], matrix1[:, 4]))
        concatenated_col2 = np.concatenate((matrix2[:, 3], matrix2[:, 4]))
        concatenated_col2_shuffled = np.concatenate((matrix2_shuffled[:, 3], matrix2_shuffled[:, 4]))

        corr_food = np.corrcoef(concatenated_col1, concatenated_col2)[0, 1]
        corr_food_shuffled = np.corrcoef(concatenated_col1, concatenated_col2_shuffled)[0, 1]


        ### explo
        concatenated_col1 = np.concatenate((matrix1[:, 5], matrix1[:, 6]))
        concatenated_col2 = np.concatenate((matrix2[:, 5], matrix2[:, 6]))
        concatenated_col2_shuffled = np.concatenate((matrix2_shuffled[:, 5], matrix2_shuffled[:, 6]))

        corr_explo = np.corrcoef(concatenated_col1, concatenated_col2)[0, 1]
        corr_explo_shuffled = np.corrcoef(concatenated_col1, concatenated_col2_shuffled)[0, 1]

        result_list.extend([corr_seq, corr_seq_shuffled, corr_zones, corr_zones_shuffled , corr_food, corr_food_shuffled, corr_explo, corr_zones_shuffled])


    total_list.append(result_list)


total_df = pd.DataFrame(total_list)
print(total_df)


means = total_df.mean()
print(means)

pos = []

for i in range(8):
    for j in range(4):
        pos.extend([0 + 3*j + 15*i,1 + 3*j + 15*i])

print(pos)

shuffle_pos = [x for x in pos if pos.index(x)%2 != 0]
true_pos = [x for x in pos if pos.index(x)%2 == 0]

color_list = ['blue', 'green', 'purple', 'lightgray']
label_list= ['Sequence', 'Zones', 'Food', 'Explo']

plt.figure(figsize=(13,3))
for i in range(total_df.shape[1]):
    col = total_df.iloc[:, i]
    col_mean = col.mean(skipna=True)
    if pos[i] in shuffle_pos:
        color = 'grey'
    else:
        color = color_list[true_pos.index(pos[i])%4]
        
    plt.bar(pos[i], col_mean, edgecolor = 'black', color = color)
    plt.scatter([pos[i]] * len(col), col, color='none', edgecolor='black', s=14)



plt.ylabel('Percentage of correlation')
plt.xticks([])
plt.tight_layout()
plt.show()


# plt.figure()
# plt.bar(1,2, color ='blue', label = 'Sequence')
# plt.bar(1,2, color ='green', label = 'Zones')
# plt.bar(1,2, color ='purple', label = 'Food')
# plt.bar(1,2, color ='lightgray', label = 'Explo')
# plt.bar(1,2, color ='grey', label = 'Shuffle')
# plt.legend()
# plt.show()