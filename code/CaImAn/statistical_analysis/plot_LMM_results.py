import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm, ols




# data = [6.922533e-50, 2.601720e-81, 2.198980e-12, 1.309690e-136, 8.694990e-135, 1.114747e-19]
# indices = np.arange(len(data))

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.bar(indices, data, align='center', alpha=0.5)

# # Set logarithmic scale
# plt.yscale('log')

# # Adding labels and title
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Logarithmic Bar Plot of Data')

# # Adding grid
# plt.grid(True)

# # Show plot
# plt.show()



big_name = 'L023'



neurons_data_df_path = f'/Users/cyrilvanleer/Desktop/Thesis/neurons/similarities_8/{big_name}/data_stat.csv'
neurons_data_df = pd.read_csv(neurons_data_df_path)

decon_data_df_path = f'/Users/cyrilvanleer/Desktop/Thesis/deconvoled/similarities_8/{big_name}/data_stat.csv'
decon_data_df = pd.read_csv(decon_data_df_path)



def lmm_analysis(data_df):


    formula = 'Frequency ~  Behavior  + Cluster + Day_of_recording + Name'

    lmm = mixedlm(formula, data_df, groups=data_df['Neuron_ID'])
    lmm_fit = lmm.fit()

    #print(lmm_fit.summary())
    #print(lmm_fit.pvalues)

    # Extract the fixed effects (intercepts and coefficients) from the LMM
    fixed_effects = lmm_fit.fe_params

    # Define your ANOVA model using the fixed effects from the LMM
    anova_formula = 'Frequency ~ Behavior + Cluster + Day_of_recording + Name'
    anova_model = ols(anova_formula, data_df).fit()

    anova_table = sm.stats.anova_lm(anova_model)
    #print(anova_table)

    p_values = anova_table['PR(>F)'].tolist()
    return(p_values[:-1])




neurons_result = lmm_analysis(neurons_data_df)
decon_result = lmm_analysis(decon_data_df)

print(neurons_result)
print(decon_result)


neuron_pos = [0, 3, 6, 9]
decon_pos = [1, 4, 7, 10]
fig, axes = plt.subplots(1, 4, figsize=(9, 4))

if big_name == 'L023':
    color_list = ['darkblue', 'darkblue']
if big_name == 'I236':
    color_list = ['indianred', 'indianred']


factor_list = ['Behavior', 'Cluster', 'Day of recording', 'Mouse']

# Plotting on each subplot
for i, (neuron_val, decon_val) in enumerate(zip(neurons_result, decon_result)):
    axes[i].bar(0, neuron_val, align='center', alpha=0.6, color = color_list[0])
    axes[i].bar(1, decon_val, align='center', alpha=1, color=color_list[1])
    axes[i].set_xticks([])
    axes[i].set_xticklabels([])

    # if i != 3:
    #     axes[i].set_yscale('log')

    axes[i].set_yscale('log')

    # Set titles for subplots
    axes[i].set_title(factor_list[i])

# Show plot
plt.tight_layout()  # Adjust layout to prevent overlapping of subplots
plt.show()

plt.figure()
plt.bar(1,0, alpha=0.6, color = color_list[0], label='CaImAn')
plt.bar(1,0, alpha=1, color = color_list[0], label='MLspike')
plt.legend()
plt.show()