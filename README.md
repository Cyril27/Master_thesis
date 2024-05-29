# Abstract
The basal ganglia and its main entry nucleus the striatum play a critical role in the control and the modulation of spontaneous and voluntary actions. 
In parallel, research has established the impact of the two efferent pathways of the striatum in motor learning. However, a more precise study of the different neurons subpopulations of both pathways and their functionalities would
enable a more accurate understanding of encoding mechanisms used by the striatum during the
learning process. Using a combination of synchronous behavioral and neuronal activity 
recordings, a suitable framework for inter-neuron and even inter-animal analysis is built, allowing
comparison of neuron dynamics and their evolution across training sessions. We demonstrated
that direct and indirect pathways offer similar overall dynamics in terms of neuron engagement
and specification. A more precise analysis revealed the existence of specific and divergent 
subpopulations in each pathway. By studying their dynamics and encoding patterns, we were able
to determine the role of each of these neuron clusters in motor learning. In addition, supervised learning algorithms allow us to extract that neuronal code is biased towards activation in
the direct pathway and likely biased towards inactivation in the indirect pathway. The results
highlight the presence of different direct and indirect pathway neurons subpopulations whose
combination enables motor learning and thus the optimal choice of behaviors to adopt for the
execution of the motor task.


# Repository structure
The repository contains all the Python code needed to process the data used, as well as its representation in various figures. A brief description of each python file is given below.

## Dat_files
**average_transition** : lalala\
**behaviors_and_candidates** : 
**behaviors_function** :
**duration_behaviors** :
**duration_cycles** :
**duration_experience** :
**get_dat** :
**get_files** :
**get_image** :
**image_function** :
**interface** :
**num_cycles** :
**num_neurons** :
**num_sessions** :
**offset_percentile** :
**quality_registration** :

## CaImAn

### Behaviors determination

**bfp** : 
**frac_of_behav** : 
**long_reg_8** : 
**opti_similarities** : 
**opti_similarities_combined** : 
**percentile** : 

### Centroids

**find_centroids** : Extract centroid coordinates from its image in the FoV\
**plot_centroids** : Replace each detect centroid in the FoV\

### Clustering

**all_clustering_methods** : Clusters pooled neurons with 9 different approaches

### Files

**get_sessions_files** : Extract different file paths for a given mouse at a given session

### Statitstical analysis

**extended_csv** : Statistical analysis over original and additional neurons\
**plot_LMM_results** : 
**prep_LMM_csv** : Prepare .csv file and apply LMM and ANOVA on it\
**stat_plots** : 

## MLspike

**compare_clusters** : 
**compare_percentiles** : 
**decon_spikes** : Compute the percentiles from spikes trains\
**find_new_candidates** : 

## Traces

**compare_scores** : Compare traces clustering MLspike and CaImAn clusterings\
**temporal_corr** : Compute temporal correlation of instants and clusters instants based on it \
**trace_percentile** : Compute the percentiles from calcium traces\

## Populations

**combined_svm** : Perform SVM on the clusters augmented by additional neurons\
**compare_pop** : Illustration differences in prediction quality and size amongst clusters\
**svm_stat** : 


