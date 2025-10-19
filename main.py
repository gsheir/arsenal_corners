import os

import matplotlib.pyplot as plt
import pandas as pd

from clustering import (cluster_corner_kmeans, get_centroid_as_corner_paths,
                        get_centroids, perform_kmeans, perform_pca)
from plotting_tools import (plot_corner_heatmap, plot_corner_paths,
                            plot_corner_zones, plot_k_means_results,
                            plot_multiple_corner_paths,
                            plot_start_end_heatmaps)
from settings import ALL_ZONES, CORNER_ZONES, OUT_CORNER_ZONES, OUTPUT_DIR
from utils import (calculate_play_quality, convert_zones_to_xy,
                   mirror_right_corners)


def create_corner_zone_plot():
    print("Creating corner zone plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    out_file = f"{OUTPUT_DIR}/corner_zones.png"
    plot_corner_zones(CORNER_ZONES, OUT_CORNER_ZONES, ax=ax, out_file=out_file)


def create_left_right_heatmaps(corners):
    print("Creating left-right heatmaps...")
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
    print("Creating start-end heatmaps...")
    for group in players["Corner group"].unique():
        out_file = f"{OUTPUT_DIR}/start_end_heatmaps_{group}.png"
        fig = plot_start_end_heatmaps(players, group, ALL_ZONES, out_file=out_file)


def create_all_corner_paths_plot(corners, players):
    print("Creating all corner paths plot...")
    out_file_prefix = f"{OUTPUT_DIR}/corner_paths"
    plot_multiple_corner_paths(corners["ID"], players, out_file_prefix=out_file_prefix)


def create_plots(corners, players):
    create_corner_zone_plot()
    create_left_right_heatmaps(corners)
    create_start_end_heatmaps(players)
    create_all_corner_paths_plot(corners, players)


def run_clustering(corners, players):
    print("Running clustering analysis...")
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
    player_paths = player_paths.drop(
        columns=["ID", "Side", "Start location", "End location"]
    )

    # Clustering run: All roles included except mop up
    print("Clustering with all roles (except mop up)...")
    player_paths = player_paths[player_paths["Role"] != "Mop up"]
    player_paths = player_paths.groupby("Corner ID").filter(lambda x: len(x) == 5)

    n_clusters = 4
    k_means_results = perform_kmeans(player_paths, n_clusters=n_clusters)
    plot_k_means_results(k_means_results, filename_prefix="all_roles_")

    # Clustering run: Shot target labelled only but pass targets are shot targets
    print("Clustering with shot target labelled only...")
    player_paths_shot_only = player_paths.copy()
    player_paths_shot_only["Role"] = player_paths_shot_only["Role"].replace(
        {"Pass target": "Shot target"}
    )
    player_paths_shot_only.loc[
        player_paths_shot_only["Role"] != "Shot target", "Role"
    ] = "Other"

    n_clusters = 4
    k_means_results = perform_kmeans(player_paths_shot_only, n_clusters=n_clusters)
    plot_k_means_results(k_means_results, filename_prefix="shot_target_")

    # Clustering run: No roles
    print("Clustering with no roles...")
    player_paths_no_roles = player_paths.copy()
    player_paths_no_roles["Role"] = "Player"

    n_clusters = 3
    k_means_results = perform_kmeans(player_paths_no_roles, n_clusters=n_clusters)
    plot_k_means_results(k_means_results, filename_prefix="no_roles_")


def run_play_quality_analysis(players):
    print("Running play quality analysis...")
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
    print("Running analysis...")

    # Clear output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    else:
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    corners = pd.read_csv("data/corners.csv")
    players = pd.read_csv("data/players.csv")

    players = pd.merge(
        left=players,
        right=corners[["Side", "ID"]].rename(columns={"ID": "Corner ID"}),
        on="Corner ID",
        how="left",
    )

    players = convert_zones_to_xy(
        players, "Start location", "start_x", "start_y", ALL_ZONES
    )
    players = convert_zones_to_xy(players, "End location", "end_x", "end_y", ALL_ZONES)
    players = mirror_right_corners(players, start_x_col="start_x", end_x_col="end_x")

    create_plots(corners, players)

    run_clustering(corners, players)

    run_play_quality_analysis(players)

    print("Done.")


if __name__ == "__main__":
    run_analysis()
