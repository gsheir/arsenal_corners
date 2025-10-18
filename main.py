import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clustering import (cluster_corner_kmeans, get_centroid_as_corner_paths,
                        get_centroids, perform_pca)
from plotting_tools import (plot_corner_heatmap, plot_corner_paths,
                            plot_corner_zones, plot_multiple_corner_paths,
                            plot_start_end_heatmaps)
from settings import ALL_ZONES, CORNER_ZONES, OUT_CORNER_ZONES, OUTPUT_DIR
from utils import (calculate_play_quality, convert_zones_to_xy,
                   mirror_right_corners)


def create_corner_zone_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    out_file = f"{OUTPUT_DIR}/corner_zones.png"
    plot_corner_zones(CORNER_ZONES, OUT_CORNER_ZONES, ax=ax, out_file=out_file)


def create_left_right_heatmaps(corners):
    left_corners = corners[corners["Side"] == "Left"]
    right_corners = corners[corners["Side"] == "Right"]
    fig, ax = plt.subplots(ncols=2, figsize=(20, 8), constrained_layout=True)

    plot_corner_heatmap(
        left_corners.groupby("Target location")["Target location"]
        .count()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False),
        col_name="Target location",
        metric_col_name="Count",
        all_zones=ALL_ZONES,
        title="AWFC corner target locations (from left)",
        ax=ax[0],
    )
    plot_corner_heatmap(
        right_corners.groupby("Target location")["Target location"]
        .count()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False),
        col_name="Target location",
        metric_col_name="Count",
        all_zones=ALL_ZONES,
        title="AWFC corner target locations (from right)",
        ax=ax[1],
    )

    out_file = f"{OUTPUT_DIR}/left_right_heatmaps.png"
    plt.savefig(out_file)


def create_start_end_heatmaps(players):
    for group in players["Corner group"].unique():
        out_file = f"{OUTPUT_DIR}/start_end_heatmaps_{group}.png"
        fig = plot_start_end_heatmaps(players, group, ALL_ZONES, out_file=out_file)


def create_all_corner_paths_plot(corners, players):
    out_file_prefix = f"{OUTPUT_DIR}/corner_paths"
    plot_multiple_corner_paths(corners["ID"], players, out_file_prefix=out_file_prefix)


def create_plots(corners, players):
    create_corner_zone_plot()
    create_left_right_heatmaps(corners)
    create_start_end_heatmaps(players)
    create_all_corner_paths_plot(corners, players)


def run_clustering(corners, players):
    # Reduce dimensions of data to only the columns we need
    player_paths = players[
        ["Start location", "End location", "Role", "Corner ID"]
    ].copy()

    player_paths = pd.merge(
        left=player_paths,
        right=corners[["Side", "ID"]],
        left_on="Corner ID",
        right_on="ID",
        how="left",
    )
    player_paths = convert_zones_to_xy(
        player_paths, "Start location", "start_x", "start_y", ALL_ZONES
    )
    player_paths = convert_zones_to_xy(
        player_paths, "End location", "end_x", "end_y", ALL_ZONES
    )
    player_paths = mirror_right_corners(
        player_paths, start_x_col="start_x", end_x_col="end_x"
    )
    player_paths = player_paths.drop(
        columns=["ID", "Side", "Start location", "End location"]
    )

    # We will assume that the "mop up" role doesn't significantly affect the clustering.
    # We will also assume the "pass target" role is functionally very similar to the "shot target" role, so we will replace it.

    player_paths = player_paths[player_paths["Role"] != "Mop up"]
    # player_paths["Role"] = player_paths["Role"].replace(
    #     {"Pass target": "Shot target"}
    # )
    player_paths = player_paths.groupby("Corner ID").filter(lambda x: len(x) == 5)

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

    out_file = f"{OUTPUT_DIR}/clustered_corner_paths.png"
    fig, ax = plt.subplots(
        nrows=n_clusters, ncols=1, figsize=(10, 5 * n_clusters), constrained_layout=True
    )
    for idx, centroid in centroids_df.iterrows():
        paths = get_centroid_as_corner_paths(centroid, centroids_df)

        # Get the corner IDs that belong to this cluster
        cluster_corners = corners_normalised[
            corners_normalised["Cluster"] == centroid["Cluster"]
        ]
        paths_df = pd.DataFrame(paths)
        plot_corner_paths(
            paths_df, title=f"Cluster {int(centroid['Cluster'])} Centroid", ax=ax[idx]
        )
        # List corner IDs in this cluster
        corner_ids = cluster_corners.index.tolist()
        ax[idx].text(
            0.5,
            -0.1,
            f"Corner IDs: {corner_ids}",
            transform=ax[idx].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    plt.savefig(out_file)

    X_pca, pca = perform_pca(X_scaled, n_components=2)

    out_file = f"{OUTPUT_DIR}/corner_clusters_pca.png"
    plt.figure(figsize=(8, 6))
    for i in range(n_clusters):
        plt.scatter(
            X_pca[corners_normalised["Cluster"] == i, 0],
            X_pca[corners_normalised["Cluster"] == i, 1],
            label=f"Cluster {i}",
        )

    # Plot centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        s=50,
        c="black",
        marker="X",
        label="Centroids",
    )

    # Add labels for each data point
    for i, corner_id in enumerate(corners_normalised.index):
        plt.text(X_pca[i, 0], X_pca[i, 1], str(corner_id), fontsize=8)

    plt.title("PCA of Corner Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(out_file)

    # For each cluster, plot play diagrams for each corner ID in a grid with 4 columns

    out_file_prefix = f"{OUTPUT_DIR}/clustered_corners"
    for cluster_id in range(n_clusters):
        corners_in_cluster = corners_normalised[
            corners_normalised["Cluster"] == cluster_id
        ]

        nrows = int(np.ceil(len(corners_in_cluster) / 4))
        fig, axs = plt.subplots(
            nrows, 4, figsize=(16, 4 * nrows), constrained_layout=True
        )
        axs = axs.flatten()

        for idx, corner_id in enumerate(corners_in_cluster.index):
            corner_paths = corners_in_cluster.loc[corner_id]
            paths = []
            for i in range(1, 6):
                path = {
                    "start_x": corner_paths[f"start_x_{i}"],
                    "start_y": corner_paths[f"start_y_{i}"],
                    "end_x": corner_paths[f"end_x_{i}"],
                    "end_y": corner_paths[f"end_y_{i}"],
                    "Role": corner_paths[f"Role_{i}"],
                }
                paths.append(path)
            paths_df = pd.DataFrame(paths)
            fig = plot_corner_paths(
                paths_df, title=f"Corner ID {corner_id}", ax=axs[idx], legend=False
            )
        plt.suptitle(f"Cluster {cluster_id} Corners", fontsize=16)
        plt.savefig(f"{out_file_prefix}_cluster_{cluster_id}.png")


def run_play_quality_analysis(players):
    players = calculate_play_quality(players)

    corner_play_quality = (
        players.groupby("Corner ID")["Play quality"]
        .sum()
        .reset_index(name="Play quality")
        .sort_values(by="Play quality", ascending=False)
    )

    print("Corner Play Quality:")
    print(corner_play_quality)

    mean_player_play_quality = (
        players.groupby("Player name")["Play quality"]
        .mean()
        .reset_index(name="Mean play quality")
        .sort_values(by="Mean play quality", ascending=False)
    )

    print("Mean Player Play Quality:")
    print(mean_player_play_quality)


def run_analysis():
    corners = pd.read_csv("data/corners.csv")
    players = pd.read_csv("data/players.csv")

    players = convert_zones_to_xy(
        players, "Start location", "start_x", "start_y", ALL_ZONES
    )
    players = convert_zones_to_xy(players, "End location", "end_x", "end_y", ALL_ZONES)

    create_plots(corners, players)

    run_clustering(corners, players)

    run_play_quality_analysis(players)


if __name__ == "__main__":
    run_analysis()
