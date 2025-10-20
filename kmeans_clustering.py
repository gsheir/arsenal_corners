import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class KMeansClustering:
    def __init__(self, player_paths, n_clusters=3, num_players=5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = None
        self.player_paths = player_paths
        self.clustering_df = None
        self.centroids_df = None
        self.num_players = num_players

    def create_clustering_df(self):
        """
        Create clustering dataframe with role aggregation.
        Each row = one corner, columns = aggregated role features
        """
        # We need to convert into a corner-oriented table, i.e. one row per corner.
        # We will end up with a wide table where there will be a set of columns for each player involved.
        self.player_paths["player_num"] = (
            self.player_paths.groupby("Corner ID").cumcount() + 1
        )

        corners_with_paths = self.player_paths.pivot(
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
            for i in range(1, self.num_players + 1):
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

        self.clustering_df = pd.DataFrame(corners_sorted).set_index("Corner ID")
        self.clustering_df = self.clustering_df.dropna()
        print(
            f"Created clustering dataframe with {len(self.clustering_df)} corners and {len(self.clustering_df.columns)} features"
        )

    def cluster_corner_kmeans(self):
        """Perform K-means clustering on role-aggregated features"""
        # Prepare data for clustering
        # Extract numeric columns (coordinates)
        self.numeric_cols = [
            col for col in self.clustering_df.columns if not col.startswith("Role_")
        ]
        X_numeric = self.clustering_df[self.numeric_cols].values

        # One-hot encode the Role columns
        role_cols = [
            col for col in self.clustering_df.columns if col.startswith("Role_")
        ]
        self.role_dummies = pd.get_dummies(
            self.clustering_df[role_cols], prefix_sep="_"
        )

        # Combine numeric and one-hot encoded features
        X_combined = np.hstack([X_numeric, self.role_dummies.values])

        # Standardize features (important for K-means)
        self.scaler = StandardScaler()
        self.x_scaled = self.scaler.fit_transform(X_combined)

        # Fit K-means with chosen number of clusters
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, init="k-means++", n_init=50, random_state=42
        )
        self.clustering_df["Cluster"] = self.kmeans.fit_predict(self.x_scaled)

    def get_centroids(self):
        """Get cluster centroids in original scale"""
        # Show cluster centroids
        centroids = self.kmeans.cluster_centers_

        # Create a DataFrame for better visualization
        centroid_features = []
        feature_names = [f"{col}" for col in self.numeric_cols] + list(
            self.role_dummies.columns
        )

        for cluster_id, centroid in enumerate(centroids):
            # Inverse transform to get back to original scale for numeric features
            centroid_original = self.scaler.inverse_transform(centroid.reshape(1, -1))[
                0
            ]

            centroid_dict = {"Cluster": cluster_id}
            for i, feature_name in enumerate(feature_names):
                centroid_dict[feature_name] = centroid_original[i]

            centroid_features.append(centroid_dict)

        self.centroids_df = pd.DataFrame(centroid_features)

    def get_reconstructed_paths_from_centroids(self):
        """
        Convert cluster centroids back to individual player path format
        for visualization purposes
        """
        reconstructed_paths = []

        for _, centroid in self.centroids_df.iterrows():
            cluster_id = centroid["Cluster"]
            for i in range(1, self.num_players + 1):
                path = {
                    "Cluster": cluster_id,
                    "start_x": centroid[f"start_x_{i}"],
                    "start_y": centroid[f"start_y_{i}"],
                    "end_x": centroid[f"end_x_{i}"],
                    "end_y": centroid[f"end_y_{i}"],
                    "Role": None,
                }

                # Determine role based on one-hot encoded features
                role_prefix = f"Role_{i}_"
                role_cols = [
                    col
                    for col in self.centroids_df.columns
                    if col.startswith(role_prefix)
                ]
                for role_col in role_cols:
                    if centroid[role_col] > 0.5:  # Assuming one-hot encoding
                        path["Role"] = role_col.replace(role_prefix, "")
                        break

                reconstructed_paths.append(path)

        self.reconstructed_paths = pd.DataFrame(reconstructed_paths)

    def perform_pca(self, n_components=2):
        """Perform PCA for visualization"""
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.x_scaled)
        self.centroids_pca = self.pca.transform(self.kmeans.cluster_centers_)

    def run(self):
        """Run the complete clustering pipeline"""
        self.create_clustering_df()
        self.cluster_corner_kmeans()
        self.get_centroids()
        self.get_reconstructed_paths_from_centroids()
        self.perform_pca()

    def get_feature_importance(self):
        """Analyze which features contribute most to cluster separation"""
        if self.centroids_df is None:
            self.get_centroids()

        # Calculate variance across clusters for each feature
        feature_names = [col for col in self.clustering_df.columns if col != "Cluster"]
        feature_variance = {}

        for feature in feature_names:
            variance = self.centroids_df[feature].var()
            feature_variance[feature] = variance

        # Sort by variance (higher variance = more important for clustering)
        sorted_features = sorted(
            feature_variance.items(), key=lambda x: x[1], reverse=True
        )

        return pd.DataFrame(
            sorted_features, columns=["Feature", "Variance_Across_Clusters"]
        )
