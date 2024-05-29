import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os 
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind




folder_path = "/Users/cyrilvanleer/Desktop/Thesis/dat_files/paths/paths_dat"

def get_files_with(folder_path,FR):
    fr_len = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and FR in file and not 'all' in file:
            df = pd.read_csv(file_path)
            fr_len.append(len(df))

    return fr_len


fr1_len = get_files_with(folder_path, 'FR1')
fr5_len = get_files_with(folder_path, 'FR5')

print(fr1_len)
print(fr5_len)


plt.scatter([0] * len(fr1_len), fr1_len, color = 'white', edgecolors='black', label='FR1')

# Box plot for FR1
plt.boxplot(fr1_len, positions=[0], widths=0.6, showmeans=True)

# Scatter plot for FR5
plt.scatter([1] * len(fr5_len), fr5_len, color = 'white', edgecolors='black', label='FR5')

# Box plot for FR5
plt.boxplot(fr5_len, positions=[1], widths=0.6, showmeans=True)

plt.xticks([0, 1], ['FR1', 'FR5'])
plt.ylabel('Number of sessions')
plt.yticks(range(int(min(fr1_len + fr5_len)), int(max(fr1_len + fr5_len)) + 1))  # Set y-ticks to integer values
plt.title('Number of sessions in FR1 and FR5')
#plt.legend()
plt.show()



print(np.mean(fr1_len))
print(np.mean(fr5_len))