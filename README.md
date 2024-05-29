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
**average_transition** : Compute the average transition matrix for each session\
**behaviors_and_candidates** : Create illustrations for behaviors determination\
**behaviors_function** : All functions required for behaviors determination\
**duration_behaviors** : Compute the mean duration of a behavior (inside a complete cycle or not)\
**duration_cycles** : Compute the duration of a motor cycle \
**duration_experience** : Extracts the mean duration of each session \
**get_dat** : Extract all the .dat files for a given mouse \
**get_files** : Store relevant files paths in .csv files\
**get_image** : Produce the behavioral time fragmentation and transition matrices images\
**image_function** : Functions needed for the transition matrix \
**interface** : Code of the Streamlit interface used to display .dat files and filter them\
**num_cycles** : Compute the number of cycles, complete cycles and presses per action sequence\
**num_neurons** : Number of neurons in each pathway per session \
**num_sessions** : Number of training sessions for each FR protocol \
**offset_percentile** : Extract the distribution of duration of a motor cycle \
**quality_registration** : Illustrates the number of sessions in which a same neuron is detected\

## CaImAn

<ins>Behaviors determination<ins>

**frac_of_behav** : Extract the significance of pathways for a behavior and a session\
**long_reg_8** : Encode the percentiles results in categorization vectors\
**opti_similarities** : Display the clustering results for a given mouse\
**opti_similarities_combined** : Display the clustering results for the combined mice of the same pathway\
**percentile** : Compute the percentiles from CaImAn spikes trains\

### <ins>Centroids<ins>

**find_centroids** : Extract centroid coordinates from its image in the FoV\
**plot_centroids** : Replace each detect centroid in the FoV\

### <ins>Clustering<ins>

**all_clustering_methods** : Clusters pooled neurons with 9 different approaches

### F<ins>iles<ins>

**get_sessions_files** : Extract different file paths for a given mouse at a given session

### <ins>Statistical analysis<ins>

**extended_csv** : Statistical analysis over original and additional neurons\
**prep_LMM_csv** : Prepare .csv file and apply LMM and ANOVA on it\
**stat_plots** : Statistical analysis over original and additional neurons\

## MLspike

**compare_clusters** : Compare clusterings using different metrics \
**compare_percentiles** : Compare percentiles values for a given behavior and session \
**decon_spikes** : Compute the percentiles from MLspike spikes trains\
**find_new_candidates** : Find for each additional neuron the closests cluster at a given session

## Traces

**compare_scores** : Compare traces clustering with MLspike and CaImAn clusterings\
**temporal_corr** : Compute temporal correlation of instants and clusters instants based on it \
**trace_percentile** : Compute the percentiles from calcium traces\

## Populations

**combined_svm** : Perform SVM on the clusters augmented by additional neurons\
**compare_pop** : Illustrates differences in prediction quality and size amongst clusters\


