import os
import matplotlib.pyplot as plt
from _clusterFunction import cluster_std_matrices, plot_clustered_std
from _auxFunction import extract_rar, select_roi
from _stdFunctions import compute_std,plot_stdMatrix
from _classificationFunctions import plot_clustered_stdMatrix,classification_SVM
plt.rcParams.update({'font.size': 18, 'lines.markersize': 15})

#%% Preparation
# Define paths
testMatrix = os.path.join('.','testMatrix.csv')
rar_path = os.path.join('.', 'sloshingVideos.rar')
extract_folder = os.path.join('.', 'sloshingVideos')
output_folder = os.path.join('.', 'std_images')
output_base_folder = os.path.join('.', 'clustered_std_images')
# Extract the .rar file if the folder does not exist or is empty
if not os.path.exists(extract_folder) or not os.listdir(extract_folder):
    extract_rar(rar_path, extract_folder)
#%% Standard Deviation Calculation
# Select ROI from the first video in the folder   
roi = select_roi(extract_folder)
timeStart,timeEnd = 0, 65 # seconds
# Compute standard deviation images for all videos in the folder
# The function will save the std images in the output folder and return a dictionary with the results
# The dictionary will contain the video name as key and a tuple (mean_std, std_img) as value
results = compute_std(extract_folder, output_folder, roi,timeStart, timeEnd)

#Plotting the standard deviation images using the testMatrix
#Reads a test matrix CSV file, adds a column with STD values from results,
#saves the updated CSV, and creates a scatter plot of the results.
plot_stdMatrix(testMatrix,results)


#%% Perform KMeans Clustering
n_clusters = 2 # Number of clusters to use for KMeans clustering
cluster_labels = cluster_std_matrices(results, n_clusters, random_state=42)
plot_clustered_std(results, cluster_labels, output_base_folder)


#%% Classification via Support Vector Machine (SVM)
gamma = 1.5 # SVM kernel coefficient
C = 1.0 # SVM regularization parameter
kernel = 'rbf'
plot_clustered_stdMatrix(testMatrix, results, cluster_labels, cluster_names=None, cluster_col='cluster')


classification_SVM(testMatrix, gamma,C,kernel)
