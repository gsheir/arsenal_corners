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


def perform_kmeans(player_paths, n_clusters=3):
    # We need to convert into a corner-oriented table, i.e. one row per corner.
    # We will end up with a wide table where there will be a set of columns for each player involved.
    player_paths["player_num"] = player_paths.groupby("Corner ID").cumcount() + 1

    corners_with_paths = player_paths.pivot(
        index="Corner ID",
        columns="player_num",
        values=["start_x", "start_y", "end_x", "end_y", "Role"],
    )

    # Flatten the multi-index columns for easier manipulation
    corners_flat = corners_with_paths.copy()
    corners_flat.columns = ["_".join(map(str, col)) for col in corners_flat.columns]

    # Reconstruct player data for each corner
    corners_sorted = []

    for corner_id in corners_with_paths.index:
        # Extract all player data for this corner
        players_data = []
        for i in range(1, 6):
            player = {
                "start_x": corners_with_paths.loc[corner_id, ("start_x", i)],
                "start_y": corners_with_paths.loc[corner_id, ("start_y", i)],
                "end_x": corners_with_paths.loc[corner_id, ("end_x", i)],
                "end_y": corners_with_paths.loc[corner_id, ("end_y", i)],
                "Role": corners_with_paths.loc[corner_id, ("Role", i)],
            }
            players_data.append(player)

        # Sort players by role, then start_x, then start_y
        players_data_sorted = sorted(
            players_data, key=lambda x: (x["Role"], x["start_x"], x["start_y"])
        )

        # Flatten sorted data into a single row
        row = {"Corner ID": corner_id}
        for i, player in enumerate(players_data_sorted, 1):
            row[f"start_x_{i}"] = player["start_x"]
            row[f"start_y_{i}"] = player["start_y"]
            row[f"end_x_{i}"] = player["end_x"]
            row[f"end_y_{i}"] = player["end_y"]
            row[f"Role_{i}"] = player["Role"]

        corners_sorted.append(row)

    corners_normalised = pd.DataFrame(corners_sorted).set_index("Corner ID")
    corners_normalised = corners_normalised.dropna()

    n_clusters = 4
    kmeans, scaler, numeric_cols, role_dummies, X_scaled = cluster_corner_kmeans(
        corners_normalised, n_clusters=n_clusters
    )
    centroids_df = get_centroids(kmeans, scaler, numeric_cols, role_dummies)

    X_pca, pca = perform_pca(X_scaled, n_components=2)

    centroids_pca = pca.transform(kmeans.cluster_centers_)

    return {
        "n_clusters": n_clusters,
        "corners_normalised": corners_normalised,
        "centroids_df": centroids_df,
        "X_pca": X_pca,
        "centroids_pca": centroids_pca,
    }


def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca
