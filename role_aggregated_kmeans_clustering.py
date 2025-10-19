import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class RoleAggregatedKMeansClustering:
    def __init__(self, player_paths, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = None
        self.player_paths = player_paths
        self.clustering_df = None
        self.centroids_df = None

    def create_clustering_df(self):
        """
        Create clustering dataframe with role aggregation.
        Each row = one corner, columns = aggregated role features
        """
        corners = self.player_paths["Corner ID"].unique()
        roles = self.player_paths["Role"].unique()

        # Validate that we have enough role diversity
        if len(roles) < 2:
            print(f"⚠️  WARNING: Only {len(roles)} unique role(s) found: {list(roles)}")
            print("   Role aggregation works best with multiple distinct roles.")
            print(
                "   Consider using a different clustering approach for single-role data."
            )

        # Create column names for aggregated features
        columns = ["Corner ID"]
        for role in roles:
            role_clean = role.replace(" ", "_").lower()
            columns.extend(
                [
                    f"{role_clean}_count",
                    f"{role_clean}_start_centroid_x",
                    f"{role_clean}_start_centroid_y",
                    f"{role_clean}_end_centroid_x",
                    f"{role_clean}_end_centroid_y",
                    f"{role_clean}_start_spread_x",  # Standard deviation
                    f"{role_clean}_start_spread_y",
                    f"{role_clean}_end_spread_x",
                    f"{role_clean}_end_spread_y",
                ]
            )

        # Initialize dataframe
        clustering_data = []

        for corner_id in corners:
            corner_players = self.player_paths[
                self.player_paths["Corner ID"] == corner_id
            ]

            # Skip corners with missing coordinate data
            if (
                corner_players[["start_x", "start_y", "end_x", "end_y"]]
                .isna()
                .any()
                .any()
            ):
                continue

            row = {"Corner ID": corner_id}

            for role in roles:
                role_clean = role.replace(" ", "_").lower()
                role_players = corner_players[corner_players["Role"] == role]

                if len(role_players) > 0:
                    # Count
                    row[f"{role_clean}_count"] = len(role_players)

                    # Centroids
                    row[f"{role_clean}_start_centroid_x"] = role_players[
                        "start_x"
                    ].mean()
                    row[f"{role_clean}_start_centroid_y"] = role_players[
                        "start_y"
                    ].mean()
                    row[f"{role_clean}_end_centroid_x"] = role_players["end_x"].mean()
                    row[f"{role_clean}_end_centroid_y"] = role_players["end_y"].mean()

                    # Spread (use 0 if only one player)
                    if len(role_players) > 1:
                        row[f"{role_clean}_start_spread_x"] = role_players[
                            "start_x"
                        ].std()
                        row[f"{role_clean}_start_spread_y"] = role_players[
                            "start_y"
                        ].std()
                        row[f"{role_clean}_end_spread_x"] = role_players["end_x"].std()
                        row[f"{role_clean}_end_spread_y"] = role_players["end_y"].std()
                    else:
                        row[f"{role_clean}_start_spread_x"] = 0
                        row[f"{role_clean}_start_spread_y"] = 0
                        row[f"{role_clean}_end_spread_x"] = 0
                        row[f"{role_clean}_end_spread_y"] = 0
                else:
                    # No players in this role - use NaN instead of 0
                    row[f"{role_clean}_count"] = 0
                    row[f"{role_clean}_start_centroid_x"] = np.nan
                    row[f"{role_clean}_start_centroid_y"] = np.nan
                    row[f"{role_clean}_end_centroid_x"] = np.nan
                    row[f"{role_clean}_end_centroid_y"] = np.nan
                    row[f"{role_clean}_start_spread_x"] = np.nan
                    row[f"{role_clean}_start_spread_y"] = np.nan
                    row[f"{role_clean}_end_spread_x"] = np.nan
                    row[f"{role_clean}_end_spread_y"] = np.nan

            clustering_data.append(row)

        self.clustering_df = pd.DataFrame(clustering_data)
        self.clustering_df = self.clustering_df.set_index("Corner ID")

        print(
            f"Created clustering dataframe with {len(self.clustering_df)} corners and {len(self.clustering_df.columns)} features"
        )

    def cluster_corner_kmeans(self):
        """Perform K-means clustering on role-aggregated features"""
        # Separate count features from coordinate features
        count_cols = [
            col for col in self.clustering_df.columns if col.endswith("_count")
        ]
        coord_cols = [
            col
            for col in self.clustering_df.columns
            if "centroid" in col or "spread" in col
        ]

        # For count features, fill NaN with 0 (no players in that role)
        clustering_features = self.clustering_df.copy()
        clustering_features[count_cols] = clustering_features[count_cols].fillna(0)

        # For coordinate features, use a more sophisticated approach:
        # Fill NaN with the mean of non-NaN values for that feature
        for col in coord_cols:
            if (
                clustering_features[col].notna().any()
            ):  # If there are any non-NaN values
                mean_val = clustering_features[col].mean()
                clustering_features[col] = clustering_features[col].fillna(mean_val)
            else:
                # If all values are NaN (role never appears), fill with 0
                clustering_features[col] = clustering_features[col].fillna(0)

        # Standardize features
        self.scaler = StandardScaler()
        self.x_scaled = self.scaler.fit_transform(clustering_features)

        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, init="k-means++", n_init=50, random_state=42
        )
        self.clustering_df["Cluster"] = self.kmeans.fit_predict(self.x_scaled)

    def get_centroids(self):
        """Get cluster centroids in original scale"""
        centroids = self.kmeans.cluster_centers_

        # Inverse transform to get back to original scale
        centroids_original = self.scaler.inverse_transform(centroids)

        # Create DataFrame
        feature_names = [col for col in self.clustering_df.columns if col != "Cluster"]
        centroid_data = []

        for cluster_id, centroid in enumerate(centroids_original):
            centroid_dict = {"Cluster": cluster_id}
            for i, feature_name in enumerate(feature_names):
                centroid_dict[feature_name] = centroid[i]
            centroid_data.append(centroid_dict)

        self.centroids_df = pd.DataFrame(centroid_data)

    def get_reconstructed_paths_from_centroids(self):
        """
        Convert cluster centroids back to individual player path format
        for visualization purposes
        """
        if self.centroids_df is None:
            self.get_centroids()

        roles = self.player_paths["Role"].unique()
        reconstructed_paths = []

        for _, centroid in self.centroids_df.iterrows():
            cluster_id = centroid["Cluster"]

            for role in roles:
                role_clean = role.replace(" ", "_").lower()

                count = max(
                    1, int(round(centroid[f"{role_clean}_count"]))
                )  # At least 1 if any

                # Only create paths for roles that have players
                if centroid[f"{role_clean}_count"] > 0.1:  # Threshold to avoid noise

                    # Get centroid coordinates
                    start_centroid_x = centroid[f"{role_clean}_start_centroid_x"]
                    start_centroid_y = centroid[f"{role_clean}_start_centroid_y"]
                    end_centroid_x = centroid[f"{role_clean}_end_centroid_x"]
                    end_centroid_y = centroid[f"{role_clean}_end_centroid_y"]

                    # Get spread for distributing multiple players
                    start_spread_x = centroid[f"{role_clean}_start_spread_x"]
                    start_spread_y = centroid[f"{role_clean}_start_spread_y"]
                    end_spread_x = centroid[f"{role_clean}_end_spread_x"]
                    end_spread_y = centroid[f"{role_clean}_end_spread_y"]

                    # Create individual player paths
                    for player_num in range(count):
                        # Distribute players around centroid using spread
                        if count == 1:
                            # Single player at centroid
                            start_x = start_centroid_x
                            start_y = start_centroid_y
                            end_x = end_centroid_x
                            end_y = end_centroid_y
                        else:
                            # Multiple players - distribute around centroid
                            offset_factor = (player_num - (count - 1) / 2) / max(
                                1, (count - 1) / 2
                            )
                            start_x = start_centroid_x + offset_factor * start_spread_x
                            start_y = start_centroid_y + offset_factor * start_spread_y
                            end_x = end_centroid_x + offset_factor * end_spread_x
                            end_y = end_centroid_y + offset_factor * end_spread_y

                        path = {
                            "Cluster": cluster_id,
                            "Role": role,
                            "Player_num": player_num + 1,
                            "start_x": start_x,
                            "start_y": start_y,
                            "end_x": end_x,
                            "end_y": end_y,
                            "Player name": f"Cluster {cluster_id} {role} {player_num + 1}",
                            "Corner ID": f"CENTROID-{cluster_id}",
                        }
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
