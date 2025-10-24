from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

from tools.settings import OUTPUT_DIR


class SimilaritySpectralClustering:
    """
    A class to perform spectral clustering using corner similarity scores.

    This class uses the SimilarityMatrix class to calculate similarities and then
    performs spectral clustering and analysis on the results.
    """

    def __init__(
        self,
        corners: pd.DataFrame,
        players: pd.DataFrame,
        similarity_matrix: np.ndarray,
        corner_ids: List[str],
    ):
        """
        Initialize the clustering system.

        Args:
            corners: DataFrame with corner information
            players: DataFrame with player movement data (should have start_x, start_y, end_x, end_y)
        """
        self.corners = corners
        self.players = players

        self.corner_ids = corner_ids
        self.n_corners = len(self.corner_ids)

        self.similarity_matrix = similarity_matrix
        self.cluster_labels = None
        self.cluster_results = None

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
        Analyse the clusters and provide explanations for similarity.

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

            # Analyse role distribution
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

            # Analyse most common patterns for important roles
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
                        else "moderately consistent"
                        if variance < 10
                        else "variable"
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

    def run(self, n_clusters: Optional[int] = None) -> Dict:
        """
        Run the complete similarity clustering pipeline.

        Args:
            n_clusters: Number of clusters (if None, will find optimal)

        Returns:
            Dictionary with complete results
        """
        print("Starting corner similarity clustering analysis...")

        self.perform_clustering(n_clusters)
        self.analyse_clusters()
        self.plot_clustered_matrix()
        self.print_cluster_explanations()

        print("\nCorner similarity clustering analysis complete!")

        return {
            "similarity_matrix": self.similarity_matrix,
            "cluster_labels": self.cluster_labels,
            "cluster_results": self.cluster_results,
            "corner_ids": self.corner_ids,
        }
