from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from settings import ALL_ZONES, OUTPUT_DIR, ROLE_WEIGHTS
from utils import convert_zones_to_xy


class CornerSimilarityClustering:
    """
    A class to calculate corner similarity scores and perform spectral clustering.

    This class handles variable numbers of players per corner, weighted role importance,
    and provides explanations for why corners are clustered together.
    """

    def __init__(self, corners: pd.DataFrame, players: pd.DataFrame):
        """
        Initialize the clustering system.

        Args:
            corners: DataFrame with corner information
            players: DataFrame with player movement data (should have start_x, start_y, end_x, end_y)
        """
        self.corners = corners
        self.players = players
        self.corner_ids = corners["ID"].tolist()
        self.n_corners = len(self.corner_ids)

        # Role hierarchy - higher values = more important
        self.role_weights = ROLE_WEIGHTS

        # Component weights for final similarity score
        self.component_weights = {
            "role_position": 0.7,  # Role-weighted position similarity
            "role_composition": 0.2,  # Role structure similarity
            "player_count": 0.1,  # Player count similarity
        }

        # Field dimensions for normalization (based on your zone structure)
        self.field_width = 80  # x-axis
        self.field_length = 120  # y-axis

        self.similarity_matrix = None
        self.cluster_labels = None
        self.cluster_results = None

    def normalize_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """Normalize coordinates to 0-1 scale."""
        norm_x = x / self.field_width
        norm_y = y / self.field_length
        return norm_x, norm_y

    def calculate_position_distance(
        self, player1: pd.Series, player2: pd.Series
    ) -> float:
        """
        Calculate 4D distance between two players (start and end positions).

        Args:
            player1, player2: Player data with start_x, start_y, end_x, end_y

        Returns:
            Normalized distance between 0 and 1
        """
        # Check for NaN values and return max distance if any are found
        coords1 = [
            player1["start_x"],
            player1["start_y"],
            player1["end_x"],
            player1["end_y"],
        ]
        coords2 = [
            player2["start_x"],
            player2["start_y"],
            player2["end_x"],
            player2["end_y"],
        ]

        if any(pd.isna(coords1)) or any(pd.isna(coords2)):
            return 1.0  # Maximum distance for missing data

        # Normalize coordinates
        start1_x, start1_y = self.normalize_coordinates(
            player1["start_x"], player1["start_y"]
        )
        end1_x, end1_y = self.normalize_coordinates(player1["end_x"], player1["end_y"])
        start2_x, start2_y = self.normalize_coordinates(
            player2["start_x"], player2["start_y"]
        )
        end2_x, end2_y = self.normalize_coordinates(player2["end_x"], player2["end_y"])

        # Check for any NaN values after normalization
        normalized_coords = [
            start1_x,
            start1_y,
            end1_x,
            end1_y,
            start2_x,
            start2_y,
            end2_x,
            end2_y,
        ]
        if any(pd.isna(normalized_coords)) or any(np.isinf(normalized_coords)):
            return 1.0  # Maximum distance for invalid coordinates

        # Calculate Euclidean distances for start and end positions
        start_distance = euclidean([start1_x, start1_y], [start2_x, start2_y])
        end_distance = euclidean([end1_x, end1_y], [end2_x, end2_y])

        # Average the distances and normalize (max possible distance is sqrt(2))
        avg_distance = (start_distance + end_distance) / 2
        normalized_distance = avg_distance / np.sqrt(2)

        return normalized_distance

    def calculate_role_position_similarity(
        self, role_players1: pd.DataFrame, role_players2: pd.DataFrame
    ) -> float:
        """
        Calculate optimal matching similarity for players in the same role.

        Args:
            role_players1, role_players2: DataFrames of players with the same role

        Returns:
            Similarity score between 0 and 1
        """
        if len(role_players1) == 0 or len(role_players2) == 0:
            return 0.0

        # Create distance matrix
        distances = []
        for _, p1 in role_players1.iterrows():
            row = []
            for _, p2 in role_players2.iterrows():
                distance = self.calculate_position_distance(p1, p2)
                row.append(distance)
            distances.append(row)

        distances = np.array(distances)

        # Handle different numbers of players using padding
        max_players = max(len(role_players1), len(role_players2))
        padded_distances = np.ones(
            (max_players, max_players)
        )  # Fill with high distance (1.0)

        rows, cols = distances.shape
        padded_distances[:rows, :cols] = distances

        # Use Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(padded_distances)

        # Calculate average similarity from matched pairs
        total_distance = padded_distances[row_indices, col_indices].sum()
        avg_distance = total_distance / max_players

        # Convert distance to similarity
        similarity = 1 - avg_distance
        return max(0, similarity)  # Ensure non-negative

    def calculate_role_composition_similarity(
        self, corner1_players: pd.DataFrame, corner2_players: pd.DataFrame
    ) -> float:
        """
        Calculate similarity based on which roles are present in each corner.

        Args:
            corner1_players, corner2_players: DataFrames of players for each corner

        Returns:
            Role composition similarity score between 0 and 1
        """
        roles1 = set(corner1_players["Role"].unique())
        roles2 = set(corner2_players["Role"].unique())

        all_roles = roles1.union(roles2)

        if len(all_roles) == 0:
            return 1.0

        # Calculate weighted Jaccard similarity
        intersection_weight = 0
        union_weight = 0

        for role in all_roles:
            weight = self.role_weights.get(role, 0.1)

            if role in roles1 and role in roles2:
                intersection_weight += weight
            union_weight += weight

        return intersection_weight / union_weight if union_weight > 0 else 0

    def calculate_player_count_similarity(
        self, corner1_players: pd.DataFrame, corner2_players: pd.DataFrame
    ) -> float:
        """
        Calculate similarity based on total number of players.

        Args:
            corner1_players, corner2_players: DataFrames of players for each corner

        Returns:
            Player count similarity score between 0 and 1
        """
        count1 = len(corner1_players)
        count2 = len(corner2_players)

        if count1 == 0 and count2 == 0:
            return 1.0

        max_count = max(count1, count2)
        min_count = min(count1, count2)

        return min_count / max_count if max_count > 0 else 0

    def calculate_corner_similarity(self, corner1_id: str, corner2_id: str) -> float:
        """
        Calculate comprehensive similarity between two corners.

        Args:
            corner1_id, corner2_id: Corner identifiers

        Returns:
            Similarity score between 0 and 1
        """
        corner1_players = self.players[self.players["Corner ID"] == corner1_id].copy()
        corner2_players = self.players[self.players["Corner ID"] == corner2_id].copy()

        if len(corner1_players) == 0 or len(corner2_players) == 0:
            return 0.0

        # Component 1: Role-weighted position similarity
        role_similarities = []
        total_role_weight = 0

        all_roles = set(corner1_players["Role"].unique()).union(
            set(corner2_players["Role"].unique())
        )

        for role in all_roles:
            role_weight = self.role_weights.get(role, 0.1)
            role1_players = corner1_players[corner1_players["Role"] == role]
            role2_players = corner2_players[corner2_players["Role"] == role]

            if len(role1_players) == 0 and len(role2_players) == 0:
                continue

            if len(role1_players) == 0 or len(role2_players) == 0:
                # Role missing in one corner - penalty proportional to importance
                role_similarity = 0.0
            else:
                role_similarity = self.calculate_role_position_similarity(
                    role1_players, role2_players
                )

            role_similarities.append(role_similarity * role_weight)
            total_role_weight += role_weight

        role_position_similarity = (
            sum(role_similarities) / total_role_weight if total_role_weight > 0 else 0
        )

        # Component 2: Role composition similarity
        role_composition_similarity = self.calculate_role_composition_similarity(
            corner1_players, corner2_players
        )

        # Component 3: Player count similarity
        player_count_similarity = self.calculate_player_count_similarity(
            corner1_players, corner2_players
        )

        # Combine components
        final_similarity = (
            self.component_weights["role_position"] * role_position_similarity
            + self.component_weights["role_composition"] * role_composition_similarity
            + self.component_weights["player_count"] * player_count_similarity
        )

        return final_similarity

    def build_similarity_matrix(self) -> np.ndarray:
        """
        Build the full similarity matrix for all corner pairs.

        Returns:
            Symmetric similarity matrix
        """
        print("Building similarity matrix...")
        print(f"Number of corners: {self.n_corners}")
        print(f"Players data shape: {self.players.shape}")
        print(f"Players columns: {list(self.players.columns)}")

        # Check for required columns
        required_cols = ["start_x", "start_y", "end_x", "end_y", "Role", "Corner ID"]
        missing_cols = [col for col in required_cols if col not in self.players.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in players data: {missing_cols}"
            )

        self.similarity_matrix = np.zeros((self.n_corners, self.n_corners))

        # Fill diagonal with 1s (perfect self-similarity)
        np.fill_diagonal(self.similarity_matrix, 1.0)

        # Calculate upper triangle (matrix is symmetric)
        for i in range(self.n_corners):
            for j in range(i + 1, self.n_corners):
                similarity = self.calculate_corner_similarity(
                    self.corner_ids[i], self.corner_ids[j]
                )
                self.similarity_matrix[i, j] = similarity
                self.similarity_matrix[j, i] = similarity  # Symmetric

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{self.n_corners} corners...")

        print(f"Similarity matrix completed. Shape: {self.similarity_matrix.shape}")
        return self.similarity_matrix

    def find_optimal_clusters(self, max_clusters: int = 10) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            max_clusters: Maximum number of clusters to test

        Returns:
            Optimal number of clusters
        """
        if self.similarity_matrix is None:
            raise ValueError("Must build similarity matrix first!")

        max_k = min(max_clusters, self.n_corners - 1)
        silhouette_scores = []

        for k in range(2, max_k + 1):
            clustering = SpectralClustering(
                n_clusters=k, affinity="precomputed", random_state=42
            )
            labels = clustering.fit_predict(self.similarity_matrix)

            # Only calculate silhouette score if we have multiple clusters
            if len(np.unique(labels)) > 1:
                # Convert similarity matrix to distance matrix for silhouette score
                distance_matrix = 1 - self.similarity_matrix
                np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0

                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                silhouette_scores.append((k, score))

        if not silhouette_scores:
            return 2  # Default fallback

        optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k

    def perform_clustering(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Perform spectral clustering on the similarity matrix.

        Args:
            n_clusters: Number of clusters (if None, will find optimal)

        Returns:
            Cluster labels array
        """
        if self.similarity_matrix is None:
            raise ValueError("Must build similarity matrix first!")

        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()

        print(f"Performing spectral clustering with {n_clusters} clusters...")

        clustering = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=42
        )

        self.cluster_labels = clustering.fit_predict(self.similarity_matrix)

        return self.cluster_labels

    def analyse_clusters(self) -> Dict:
        """
        analyse the clusters and provide explanations for similarity.

        Returns:
            Dictionary with cluster analysis results
        """
        if self.cluster_labels is None:
            raise ValueError("Must perform clustering first!")

        results = {}
        unique_labels = np.unique(self.cluster_labels)

        for cluster_id in unique_labels:
            cluster_corner_indices = np.where(self.cluster_labels == cluster_id)[0]
            cluster_corner_ids = [self.corner_ids[i] for i in cluster_corner_indices]

            # Get all players in this cluster
            cluster_players = self.players[
                self.players["Corner ID"].isin(cluster_corner_ids)
            ].copy()

            # analyse role distribution
            role_stats = (
                cluster_players.groupby("Role")
                .agg(
                    {
                        "start_x": ["mean", "std"],
                        "start_y": ["mean", "std"],
                        "end_x": ["mean", "std"],
                        "end_y": ["mean", "std"],
                    }
                )
                .round(2)
            )

            # analyse most common patterns for important roles
            important_roles = ["Shot target", "Pass target", "Second target"]
            role_patterns = {}

            for role in important_roles:
                role_data = cluster_players[cluster_players["Role"] == role]
                if len(role_data) > 0:
                    pattern = {
                        "count": len(role_data),
                        "avg_start_pos": (
                            role_data["start_x"].mean(),
                            role_data["start_y"].mean(),
                        ),
                        "avg_end_pos": (
                            role_data["end_x"].mean(),
                            role_data["end_y"].mean(),
                        ),
                        "position_variance": (
                            role_data["start_x"].std()
                            + role_data["start_y"].std()
                            + role_data["end_x"].std()
                            + role_data["end_y"].std()
                        )
                        / 4,
                    }
                    role_patterns[role] = pattern

            results[cluster_id] = {
                "corner_ids": cluster_corner_ids,
                "size": len(cluster_corner_ids),
                "role_patterns": role_patterns,
                "role_stats": role_stats,
            }

        self.cluster_results = results
        return results

    def print_cluster_explanations(self):
        """Print human-readable explanations of why corners are clustered together."""
        if self.cluster_results is None:
            raise ValueError("Must analyse clusters first!")

        print("\n" + "=" * 80)
        print("CORNER CLUSTER ANALYSIS")
        print("=" * 80)

        for cluster_id, data in self.cluster_results.items():
            print(f"\nCLUSTER {cluster_id} ({data['size']} corners)")
            print("-" * 40)
            print(f"Corner IDs: {', '.join(data['corner_ids'])}")

            print("\nKey similarities:")

            # Explain based on role patterns
            for role, pattern in data["role_patterns"].items():
                if pattern["count"] > 0:
                    avg_start = pattern["avg_start_pos"]
                    avg_end = pattern["avg_end_pos"]
                    variance = pattern["position_variance"]

                    consistency = (
                        "highly consistent"
                        if variance < 5
                        else "moderately consistent" if variance < 10 else "variable"
                    )

                    print(f"  • {role} players ({pattern['count']} total):")
                    print(
                        f"    - Start positions: ({avg_start[0]:.1f}, {avg_start[1]:.1f}) - {consistency}"
                    )
                    print(f"    - End positions: ({avg_end[0]:.1f}, {avg_end[1]:.1f})")

            if not data["role_patterns"]:
                print(
                    "  • Primarily similar due to role composition and player count patterns"
                )

    def plot_clustered_matrix(self, figsize: Tuple[int, int] = (15, 6)):
        """
        Plot similarity matrix reordered by clusters.

        Args:
            figsize: Figure size tuple
        """
        if self.similarity_matrix is None or self.cluster_labels is None:
            raise ValueError(
                "Must build similarity matrix and perform clustering first!"
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Original similarity matrix
        sns.heatmap(
            self.similarity_matrix,
            annot=False,
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
            ax=ax1,
            cbar_kws={"label": "Similarity"},
        )
        ax1.set_title("Original Similarity Matrix")
        ax1.set_xlabel("Corner Index")
        ax1.set_ylabel("Corner Index")

        # Plot 2: Reordered by clusters
        cluster_order = np.argsort(self.cluster_labels)
        reordered_matrix = self.similarity_matrix[cluster_order][:, cluster_order]
        reordered_labels = self.cluster_labels[cluster_order]
        reordered_ids = [self.corner_ids[i] for i in cluster_order]

        sns.heatmap(
            reordered_matrix,
            annot=False,
            cmap="viridis",
            vmin=0,
            vmax=1,
            square=True,
            ax=ax2,
            cbar_kws={"label": "Similarity"},
        )

        # Add cluster boundaries
        cluster_boundaries = np.where(np.diff(reordered_labels))[0] + 0.5
        for boundary in cluster_boundaries:
            ax2.axhline(boundary, color="red", linewidth=2)
            ax2.axvline(boundary, color="red", linewidth=2)

        ax2.set_title("Clustered Similarity Matrix")
        ax2.set_xlabel("Corner Index (reordered)")
        ax2.set_ylabel("Corner Index (reordered)")

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/corner_similarity_clustering.png",
            dpi=300,
            bbox_inches="tight",
        )

    def run_complete_analysis(self, n_clusters: Optional[int] = None) -> Dict:
        """
        Run the complete similarity clustering pipeline.

        Args:
            n_clusters: Number of clusters (if None, will find optimal)

        Returns:
            Dictionary with complete results
        """
        print("Starting corner similarity clustering analysis...")

        # Step 1: Build similarity matrix
        self.build_similarity_matrix()

        # Step 2: Perform clustering
        self.perform_clustering(n_clusters)

        # Step 3: analyse clusters
        self.analyse_clusters()

        # Step 4: Create visualizations
        self.plot_clustered_matrix()

        # Step 5: Print explanations
        self.print_cluster_explanations()

        print("\nCorner similarity clustering analysis complete!")

        return {
            "similarity_matrix": self.similarity_matrix,
            "cluster_labels": self.cluster_labels,
            "cluster_results": self.cluster_results,
            "corner_ids": self.corner_ids,
        }
