from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

from tools.settings import ROLE_WEIGHTS, SIMILARITY_SETTINGS


class SimilarityCalculator:
    """
    A class to calculate corner similarity scores and build similarity matrices.

    This class handles variable numbers of players per corner, weighted role importance,
    and provides different similarity calculation methods.
    """

    def __init__(
        self,
        corners: pd.DataFrame,
        players: pd.DataFrame,
        similarity_settings: dict = SIMILARITY_SETTINGS,
    ):
        """
        Initialize the similarity matrix calculator.

        Args:
            corners: DataFrame with corner information
            players: DataFrame with player movement data (should have start_x, start_y, end_x, end_y)
        """
        self.corners = corners
        self.players = players
        self.corner_ids = corners["ID"].tolist()
        self.n_corners = len(self.corner_ids)
        self.similarity_settings = similarity_settings

        # Role hierarchy - higher values = more important
        self.role_weights = ROLE_WEIGHTS

        # Component weights for final similarity score
        self.component_weights = self.similarity_settings["component_weights"]

        # Field dimensions for normalization (based on your zone structure)
        self.field_width = 80  # x-axis
        self.field_length = 120  # y-axis

        self.similarity_matrix = None

    def normalise_coordinates(self, x: float, y: float) -> Tuple[float, float]:
        """normalise coordinates to 0-1 scale."""
        norm_x = x / self.field_width
        norm_y = y / self.field_length
        return norm_x, norm_y

    def calculate_path_similarity(
        self, player_1: pd.Series, player_2: pd.Series, method: str = "euclidean"
    ) -> float:
        """
        Calculate 4D distance between two players (start and end positions).

        Args:
            player1, player2: Player data with start_x, start_y, end_x, end_y

        Returns:
            normalised distance between 0 and 1
        """
        # Check for NaN values and return max distance if any are found
        coords1 = [
            player_1["start_x"],
            player_1["start_y"],
            player_1["end_x"],
            player_1["end_y"],
        ]
        coords2 = [
            player_2["start_x"],
            player_2["start_y"],
            player_2["end_x"],
            player_2["end_y"],
        ]

        if any(pd.isna(coords1)) or any(pd.isna(coords2)):
            return 0.0  # Zero similarity for missing data

        # normalise coordinates
        start1_x, start1_y = self.normalise_coordinates(
            player_1["start_x"], player_1["start_y"]
        )
        end1_x, end1_y = self.normalise_coordinates(
            player_1["end_x"], player_1["end_y"]
        )
        start2_x, start2_y = self.normalise_coordinates(
            player_2["start_x"], player_2["start_y"]
        )
        end2_x, end2_y = self.normalise_coordinates(
            player_2["end_x"], player_2["end_y"]
        )

        # Check for any NaN values after normalization
        normalised_coords = [
            start1_x,
            start1_y,
            end1_x,
            end1_y,
            start2_x,
            start2_y,
            end2_x,
            end2_y,
        ]
        if any(pd.isna(normalised_coords)) or any(np.isinf(normalised_coords)):
            return 0.0  # Zero similarity for missing or invalid data

        if method == "euclidean":
            # Calculate Euclidean distances for start and end positions
            start_distance = euclidean([start1_x, start1_y], [start2_x, start2_y])
            end_distance = euclidean([end1_x, end1_y], [end2_x, end2_y])

            # Average the distances and normalise (max possible distance is sqrt(2))
            avg_distance = (start_distance + end_distance) / 2
            normalised_distance = avg_distance / np.sqrt(2)

            path_similarity = 1 - normalised_distance  # Convert to similarity

        elif method == "decomposed":
            # Component 1: Start position distance
            start_distance = euclidean((start1_x, start1_y), (start2_x, start2_y))
            start_similarity = 1 - start_distance / np.sqrt(2)

            # Component 2: End position distance
            end_distance = euclidean((end1_x, end1_y), (end2_x, end2_y))
            end_similarity = 1 - end_distance / np.sqrt(2)

            # Component 3: Path length difference
            path1_length = euclidean((start1_x, start1_y), (end1_x, end1_y))
            path2_length = euclidean((start2_x, start2_y), (end2_x, end2_y))

            if path1_length + path2_length == 0:
                length_similarity = 1.0  # Both paths are zero-length
            else:
                length_similarity = 1 - abs(path1_length - path2_length) / (
                    path1_length + path2_length
                )

            # Component 4: Path direction difference
            if path1_length > 0 and path2_length > 0:
                path1_direction = (
                    (end1_x - start1_x) / path1_length,
                    (end1_y - start1_y) / path1_length,
                )
                path2_direction = (
                    (end2_x - start2_x) / path2_length,
                    (end2_y - start2_y) / path2_length,
                )
                direction_similarity = np.dot(
                    path1_direction, path2_direction
                )  # Cosine similarity
                direction_similarity = (
                    direction_similarity + 1
                ) / 2  # Normalise to [0,1]

            elif path1_length == 0 and path2_length == 0:
                direction_similarity = 1.0  # Both paths are zero-length
            else:
                direction_similarity = 0.0  # One path is zero-length

            path_similarity = (
                self.similarity_settings["path_similarity_weights"]["start_similarity"]
                * start_similarity
                + self.similarity_settings["path_similarity_weights"]["end_similarity"]
                * end_similarity
                + self.similarity_settings["path_similarity_weights"][
                    "length_similarity"
                ]
                * length_similarity
                + self.similarity_settings["path_similarity_weights"][
                    "direction_similarity"
                ]
                * direction_similarity
            )

        return path_similarity

    def calculate_role_weighted_path_similarity(
        self, role_players_1: pd.DataFrame, role_players_2: pd.DataFrame
    ) -> float:
        """
        Calculate optimal matching similarity for players in the same role.

        Args:
            role_players_1, role_players_2: DataFrames of players with the same role

        Returns:
            Similarity score between 0 and 1
        """
        if len(role_players_1) == 0 or len(role_players_2) == 0:
            return 0.0

        # Create distance matrix
        distances = []
        for _, p1 in role_players_1.iterrows():
            row = []
            for _, p2 in role_players_2.iterrows():
                distance = self.calculate_path_similarity(p1, p2, method="decomposed")
                row.append(distance)
            distances.append(row)

        distances = np.array(distances)

        # Handle different numbers of players using padding
        max_players = max(len(role_players_1), len(role_players_2))
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
        self, corner_1_players: pd.DataFrame, corner_2_players: pd.DataFrame
    ) -> float:
        """
        Calculate similarity based on which roles are present in each corner.

        Args:
            corner_1_players, corner_2_players: DataFrames of players for each corner

        Returns:
            Role composition similarity score between 0 and 1
        """
        roles1 = set(corner_1_players["Role"].unique())
        roles2 = set(corner_2_players["Role"].unique())

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
        self, corner_1_players: pd.DataFrame, corner_2_players: pd.DataFrame
    ) -> float:
        """
        Calculate similarity based on total number of players.

        Args:
            corner_1_players, corner_2_players: DataFrames of players for each corner

        Returns:
            Player count similarity score between 0 and 1
        """
        count1 = len(corner_1_players)
        count2 = len(corner_2_players)

        if count1 == 0 and count2 == 0:
            return 1.0

        max_count = max(count1, count2)
        min_count = min(count1, count2)

        return min_count / max_count if max_count > 0 else 0

    def calculate_corner_similarity(self, corner_1_id: str, corner_2_id: str) -> float:
        """
        Calculate comprehensive similarity between two corners.

        Args:
            corner_1_id, corner_2_id: Corner identifiers

        Returns:
            Similarity score between 0 and 1
        """
        corner_1_players = self.players[self.players["Corner ID"] == corner_1_id].copy()
        corner_2_players = self.players[self.players["Corner ID"] == corner_2_id].copy()

        if len(corner_1_players) == 0 or len(corner_2_players) == 0:
            return 0.0

        # Component 1: Role-weighted position similarity
        role_similarities = []
        total_role_weight = 0

        all_roles = set(corner_1_players["Role"].unique()).union(
            set(corner_2_players["Role"].unique())
        )

        for role in all_roles:
            role_weight = self.role_weights.get(role, 0.1)
            role_players_1 = corner_1_players[corner_1_players["Role"] == role]
            role_players_2 = corner_2_players[corner_2_players["Role"] == role]

            if len(role_players_1) == 0 and len(role_players_2) == 0:
                continue

            if len(role_players_1) == 0 or len(role_players_2) == 0:
                # Role missing in one corner
                role_similarity = 0.0
            else:
                role_similarity = self.calculate_role_weighted_path_similarity(
                    role_players_1, role_players_2
                )

            role_similarities.append(role_similarity * role_weight)
            total_role_weight += role_weight

        role_position_similarity = (
            sum(role_similarities) / total_role_weight if total_role_weight > 0 else 0
        )

        # Component 2: Role composition similarity
        role_composition_similarity = self.calculate_role_composition_similarity(
            corner_1_players, corner_2_players
        )

        # Component 3: Player count similarity
        player_count_similarity = self.calculate_player_count_similarity(
            corner_1_players, corner_2_players
        )

        # Combine components
        final_similarity = (
            self.component_weights["role_weighted_path_similarity"]
            * role_position_similarity
            + self.component_weights["role_composition_similarity"]
            * role_composition_similarity
            + self.component_weights["player_count_similarity"]
            * player_count_similarity
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

    def get_similarity_matrix(self) -> np.ndarray:
        """
        Get the similarity matrix, building it if necessary.

        Returns:
            Similarity matrix
        """
        if self.similarity_matrix is None:
            self.build_similarity_matrix()

        return self.similarity_matrix

    def get_corner_ids(self) -> list:
        """Get the list of corner IDs corresponding to matrix rows/columns."""
        return self.corner_ids
