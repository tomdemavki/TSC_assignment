from sklearn.cluster import KMeans
import numpy as np
import os
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def cluster_std_matrices(results, n_clusters=3, random_state=42):
    """
    Cluster the std matrices in the results dictionary.
    Returns a dict: {video_name: cluster_label}
    If output_base_folder is provided, saves std images in cluster folders.
    """
    # Prepare data: flatten each std matrix, ensuring same shape
    names = []
    mats = []
    # Find minimum shape among all std matrices to ensure consistent flattening
    shapes = [std_mat.shape for _, std_mat in results.values()]
    min_shape = (min(s[0] for s in shapes), min(s[1] for s in shapes))
    for name, (mean_std, std_mat) in results.items():
        # Crop each std matrix to the minimum shape if needed
        cropped = std_mat[:min_shape[0], :min_shape[1]]
        names.append(name)
        mats.append(cropped.flatten())
    # Stack all flattened matrices into a 2D array for clustering
    mats = np.stack(mats, axis=0)
    # Create and fit KMeans model
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(mats)
    # Map each video name to its cluster label
    cluster_labels = dict(zip(names, labels))       
    # plot the silhouette plot for the cluster
    plot_silhouette(mats)
    return cluster_labels

def plot_clustered_std(results, cluster_labels, output_base_folder):
    """
    For each cluster, create a folder and save the std image for each video in that cluster.
    Args:
        results: dict {video_name: (mean_std, std_img)}
        cluster_labels: dict {video_name: cluster_label}
        output_base_folder: str, path to base output folder
    """
    # Ensure the base output folder exists
    os.makedirs(output_base_folder, exist_ok=True)
    # Group videos by cluster label
    clusters = {}
    for name, label in cluster_labels.items():
        clusters.setdefault(label, []).append(name)
    # For each cluster, create a subfolder and save images
    for label, names in clusters.items():
        cluster_folder = os.path.join(output_base_folder, f"cluster_{label}")
        os.makedirs(cluster_folder, exist_ok=True)
        for name in names:
            mean_std, std_img = results[name]
            # Create a figure for the std image
            plt.figure(figsize=(6, 5))
            plt.imshow(std_img, cmap='inferno')
            plt.title(f"{name} (mean std: {mean_std:.3f})")
            plt.axis('off')
            plt.colorbar()
            plt.tight_layout()
            # Save the std image to the cluster folder
            plt.savefig(os.path.join(cluster_folder, f"{name}_std.png"))
            plt.close()


def plot_silhouette(mats):
    range_n_clusters = [2]
    X=mats
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(9, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    

    plt.show()


