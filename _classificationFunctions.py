import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def plot_clustered_stdMatrix(testMatrix, results, cluster_labels, cluster_names=None, cluster_col='cluster'):
    """
    Update testMatrix CSV with cluster labels and plot std values colored by cluster.
    Args:
        testMatrix: path to CSV
        results: dict {video_name: (mean_std, std_img)}
        cluster_labels: dict {video_name: cluster_label}
        cluster_names: dict {cluster_label: str}, optional mapping of cluster index to name
        cluster_col: str, name of the column to add for cluster
    """
    # Read the test matrix CSV file
    df = pd.read_csv(testMatrix)
    std_values = []
    cluster_value = []
    # For each test, get the corresponding mean_std and cluster label
    for name in df['testName']:
        if name in results:
            std_val = results[name][0]
        else:
            std_val = "ND"
        std_values.append(std_val)
        if name in cluster_labels:
            label = cluster_labels[name]
            # Write only the integer label (0, 1, ...) not "cluster_0"
            cluster_value.append(label)
        else:
            cluster_value.append("ND")
    # Add std and cluster columns to the dataframe
    df['std'] = std_values
    df[cluster_col] = cluster_value
    # Save the updated dataframe back to the CSV file
    df.to_csv(testMatrix, index=False)

    # Plotting
    # Extract x, y, z, and cluster values for plotting
    x = df['omegaNDExp'].values
    y = df['Z0/r'].values
    z = df['std'].values
    clusters = df[cluster_col].values
    # Convert std values to numeric, mask out invalid entries
    z_numeric = pd.to_numeric(z, errors='coerce')
    mask = ~np.isnan(z_numeric)
    x = x[mask]
    y = y[mask]
    z = z_numeric[mask]
    clusters = clusters[mask]

    # Assign a color to each cluster
    unique_clusters = np.unique(clusters)
    colors = plt.cm.tab10.colors
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

    plt.figure(figsize=(8, 6))
    # Plot each cluster with its own color and label
    for c in unique_clusters:
        idx = clusters == c
        plt.scatter(
            x[idx], y[idx],
            c=[color_map[c]],
            s=(z[idx] - z.min()) / (z.max() - z.min()) * 200 + 20,
            label=str(c),
            alpha=0.8, edgecolors='k', marker='o'
        )
    plt.xlabel('$\\omega_f/\\omega_n$')
    plt.ylabel('$Z_0/r$')
    plt.title("STD clusterized")
    plt.legend(title='Cluster',loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.show()

def classification_SVM(testMatrix,gamma,C,kernel):
    """
    Read testMatrix.csv, use omegaNDExp as x, Z0/r as y, and cluster as label.
    Fit an SVM and plot decision boundaries between clusters.
    Plots both in normalized and original feature space, including boundaries in the original space.

    Returns:
        clf: trained SVC object (on normalized data)
        scaler: StandardScaler object used for normalization
  
    """
    # Read the CSV file
    df = pd.read_csv(testMatrix)
    # Drop rows with missing cluster or features
    mask = df['cluster'].notna() & df['omegaNDExp'].notna() & df['Z0/r'].notna()
    df = df[mask]
    # Extract features and labels
    X = df[['omegaNDExp', 'Z0/r']].values
    y = df['cluster'].values

    # Plot original data points colored by cluster
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_xlabel('$\\omega_f/\\omega_n$')
    ax.set_ylabel('$Z_0/r$')
    ax.set_title('Original Data')
    plt.show()

    # --- Decision boundary in normalized space ---
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.3, random_state=42)

    # Create mesh grid for plotting decision boundaries
    h = 0.2
    x_min, x_max = X_scaled[:, 0].min() - .5, X_scaled[:, 0].max() + .5
    y_min, y_max = X_scaled[:, 1].min() - .5, X_scaled[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Train SVM classifier on normalized data
    clf = SVC(C=C, kernel=kernel, gamma=gamma)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("SVM accuracy (normalized):", score)

    # Compute decision function for mesh grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cm = plt.cm.RdBu

    # Plot decision boundaries and data in normalized space
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    ax.set_xlabel('Normalized $\\omega_f/\\omega_n$')
    ax.set_ylabel('Normalized $Z_0/r$')
    ax.set_title('SVM Decision Boundaries (normalized)')
    plt.show()

    # --- Decision boundary in original space using SVM trained on normalized data ---
    # Create a mesh in the original feature space
    h = 0.01
    x_min_orig, x_max_orig = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min_orig, y_max_orig = X[:, 1].min() - .2, X[:, 1].max() + .2
    xx_orig, yy_orig = np.meshgrid(np.arange(x_min_orig, x_max_orig, h),
                                   np.arange(y_min_orig, y_max_orig, h))
    # Transform mesh points to normalized space
    mesh_points = np.c_[xx_orig.ravel(), yy_orig.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    # Compute decision function for mesh in original space
    Z_orig = clf.decision_function(mesh_points_scaled)
    Z_orig = Z_orig.reshape(xx_orig.shape)

    # Plot decision boundaries and data in original feature space
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.contourf(xx_orig, yy_orig, Z_orig, cmap=cm, alpha=.5)
    ax.set_xlabel('$\\omega_f/\\omega_n$')
    ax.set_ylabel('$Z_0/r$')
    ax.set_title('SVM Decision Boundaries')
    ax.set_xlim(0.88,1.22)
    ax.set_ylim(0,0.4)
    plt.grid()
    plt.show()

    return clf, scaler
