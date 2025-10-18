import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def cluster_corner_kmeans(corners_normalised, n_clusters=3):
    # Prepare data for clustering
    # Extract numeric columns (coordinates)
    numeric_cols = [
        col for col in corners_normalised.columns if not col.startswith("Role_")
    ]
    X_numeric = corners_normalised[numeric_cols].values

    # One-hot encode the Role columns
    role_cols = [col for col in corners_normalised.columns if col.startswith("Role_")]
    role_dummies = pd.get_dummies(corners_normalised[role_cols], prefix_sep="_")

    # Combine numeric and one-hot encoded features
    X_combined = np.hstack([X_numeric, role_dummies.values])

    # Standardize features (important for K-means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Fit K-means with chosen number of clusters
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=50, random_state=42)
    corners_normalised["Cluster"] = kmeans.fit_predict(X_scaled)

    return kmeans, scaler, numeric_cols, role_dummies, X_scaled


def get_centroids(kmeans, scaler, numeric_cols, role_dummies):
    # Show cluster centroids
    centroids = kmeans.cluster_centers_

    # Create a DataFrame for better visualization
    centroid_features = []
    feature_names = [f"{col}" for col in numeric_cols] + list(role_dummies.columns)

    for cluster_id, centroid in enumerate(centroids):
        # Inverse transform to get back to original scale for numeric features
        centroid_original = scaler.inverse_transform(centroid.reshape(1, -1))[0]

        centroid_dict = {"Cluster": cluster_id}
        for i, feature_name in enumerate(feature_names):
            centroid_dict[feature_name] = centroid_original[i]

        centroid_features.append(centroid_dict)

    centroids_df = pd.DataFrame(centroid_features)

    return centroids_df


def get_centroid_as_corner_paths(centroid, centroids_df):
    paths = []

    for i in range(1, 6):
        path = {
            "start_x": centroid[f"start_x_{i}"],
            "start_y": centroid[f"start_y_{i}"],
            "end_x": centroid[f"end_x_{i}"],
            "end_y": centroid[f"end_y_{i}"],
            "Role": None,
        }

        # Determine role based on one-hot encoded features
        role_prefix = f"Role_{i}_"
        role_cols = [col for col in centroids_df.columns if col.startswith(role_prefix)]
        for role_col in role_cols:
            if centroid[role_col] > 0.5:  # Assuming one-hot encoding
                path["Role"] = role_col.replace(role_prefix, "")
                break

        paths.append(path)

    return paths


def perform_pca(X, n_components=2):

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
