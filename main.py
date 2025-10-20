import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corner_similarity_clustering import CornerSimilarityClustering
from kmeans_clustering import KMeansClustering
from plotting_tools import (plot_corner_heatmap, plot_corner_paths,
                            plot_corner_zones, plot_k_means_results,
                            plot_multiple_corner_paths,
                            plot_start_end_heatmaps)
from role_aggregated_kmeans_clustering import RoleAggregatedKMeansClustering
from settings import ALL_ZONES, CORNER_ZONES, OUT_CORNER_ZONES, OUTPUT_DIR
from utils import (add_play_quality_to_players, convert_zones_to_xy,
                   get_mean_play_quality_for_corner_ids,
                   mirror_right_corners_for_corners,
                   mirror_right_corners_for_players)


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


def create_left_front_post_delivery_vs_run_heatmap(corners, players):
    print("Creating left front post delivery vs run heatmap...")
    group_corners = corners[
        corners["Group"].isin(
            ["Front post multiple shot targets", "Front post single shot target"]
        )
    ]
    corner_players = players[
        players["Group"].isin(
            ["Front post multiple shot targets", "Front post single shot target"]
        )
        & players["Role"].isin(["Shot target"])
    ]

    fig, ax = plt.subplots(ncols=2, figsize=(20, 8), constrained_layout=True)
    plot_corner_heatmap(
        group_corners.groupby("Target location")["Target location"]
        .count()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False),
        col_name="Target location",
        metric_col_name="Count",
        all_zones=ALL_ZONES,
        title="Corner target locations (front post delivery)",
        ax=ax[0],
    )
    plot_corner_heatmap(
        corner_players.groupby("End location")["End location"]
        .count()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False),
        col_name="End location",
        metric_col_name="Count",
        all_zones=ALL_ZONES,
        title="Player end locations (front post delivery)",
        ax=ax[1],
    )

    out_file = f"{OUTPUT_DIR}/left_front_post_delivery_vs_run_heatmap.png"
    plt.savefig(out_file)


def create_all_corner_paths_plot(corners, players):
    print("Creating all corner paths plot...")
    out_file_prefix = f"{OUTPUT_DIR}/corner_paths"
    plot_multiple_corner_paths(corners["ID"], players, out_file_prefix=out_file_prefix)


def create_hand_clustered_corner_paths_plot(corners, players):
    print("Creating hand-clustered corner paths plot...")
    for group in corners["Group"].unique():
        clustered_corner_ids = corners[corners["Group"] == group]["ID"].tolist()

        out_file_prefix = f"{OUTPUT_DIR}/hand_clustered_corner_paths_{group.replace(' ', '_').lower()}.png"
        mean_play_quality = get_mean_play_quality_for_corner_ids(
            players, clustered_corner_ids
        )
        plot_multiple_corner_paths(
            clustered_corner_ids,
            players,
            out_file_prefix=out_file_prefix,
            title=f"Corner paths - {group} (Mean Play Quality: {mean_play_quality:.3f})",
        )


def create_plots(corners, players):
    create_corner_zone_plot()
    create_left_right_heatmaps(corners)
    create_start_end_heatmaps(players)
    create_all_corner_paths_plot(corners, players)
    create_hand_clustered_corner_paths_plot(corners, players)
    create_left_front_post_delivery_vs_run_heatmap(corners, players)


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
    player_paths = player_paths[player_paths["Role"] != "Mop up"]

    # Clustering run: All roles included except mop up
    print("Clustering with all roles (except mop up)...")
    player_paths_all_roles = player_paths.copy()

    n_clusters = 5
    clustering = RoleAggregatedKMeansClustering(
        player_paths_all_roles, n_clusters=n_clusters
    )
    clustering.run()
    plot_k_means_results(clustering, players, filename_prefix="all_roles_")

    # Clustering run: Shot target labelled only but pass targets are shot targets
    print("Clustering with shot target labelled only...")
    player_paths_shot_only = player_paths.copy()
    player_paths_shot_only["Role"] = player_paths_shot_only["Role"].replace(
        {"Pass target": "Shot target"}
    )
    player_paths_shot_only.loc[
        player_paths_shot_only["Role"] != "Shot target", "Role"
    ] = "Other"
    player_paths_shot_only = player_paths_shot_only.groupby("Corner ID").filter(
        lambda x: len(x) == 5
    )

    n_clusters = 4
    clustering = KMeansClustering(player_paths_shot_only, n_clusters=n_clusters)
    clustering.run()
    plot_k_means_results(clustering, players, filename_prefix="shot_target_aware_")

    # Clustering run: No roles
    print("Clustering with no roles...")
    player_paths_no_roles = player_paths.copy()
    player_paths_no_roles["Role"] = "Player"
    player_paths_no_roles = player_paths_no_roles.groupby("Corner ID").filter(
        lambda x: len(x) == 5
    )

    n_clusters = 3
    clustering = KMeansClustering(player_paths_no_roles, n_clusters=n_clusters)
    clustering.run()
    plot_k_means_results(clustering, players, filename_prefix="no_roles_")


def run_similarity_clustering(corners, players):
    """Run similarity-based clustering analysis on corners."""
    print("Running similarity-based clustering analysis...")

    # Create and run the similarity clustering
    similarity_clustering = CornerSimilarityClustering(corners, players)
    results = similarity_clustering.run_complete_analysis()

    return results


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
        right=corners[["Side", "ID", "Group", "Discard"]].rename(
            columns={"ID": "Corner ID"}
        ),
        on="Corner ID",
        how="left",
    )

    corners = corners[corners["Discard"] == "No"]
    corners = mirror_right_corners_for_corners(corners)

    players = players[players["Discard"] == "No"]
    players = convert_zones_to_xy(
        players, "Start location", "start_x", "start_y", ALL_ZONES
    )
    players = convert_zones_to_xy(players, "End location", "end_x", "end_y", ALL_ZONES)
    players = mirror_right_corners_for_players(
        players, start_x_col="start_x", end_x_col="end_x"
    )
    players = add_play_quality_to_players(players)

    create_plots(corners, players)

    # Print hand clustered corner group mean play qualities
    front_post_mean_play_quality = get_mean_play_quality_for_corner_ids(
        players,
        corners[
            corners["Group"].isin(
                [
                    "Front post multiple shot targets",
                    "Front post single shot target",
                ]
            )
        ]["ID"].tolist(),
    )
    print(f"Mean Play Quality for Front Post Group: {front_post_mean_play_quality:.3f}")

    # Print hand clustered corner group mean play qualities
    back_post_mean_play_quality = get_mean_play_quality_for_corner_ids(
        players,
        corners[corners["Group"].isin(["Back post with decoy to front"])][
            "ID"
        ].tolist(),
    )
    print(f"Mean Play Quality for Back Post Group: {back_post_mean_play_quality:.3f}")

    short_lead_mean_play_quality = get_mean_play_quality_for_corner_ids(
        players,
        corners[corners["Group"].isin(["Short lead"])]["ID"].tolist(),
    )
    print(f"Mean Play Quality for Short Lead Group: {short_lead_mean_play_quality:.3f}")

    # Rank corners by play quality
    corners_play_quality = (
        players.groupby("Corner ID")["Play quality"].sum().reset_index()
    )
    corners_play_quality = pd.merge(
        left=corners_play_quality,
        right=corners[["ID", "Group", "Game", "Minute"]].rename(
            columns={"ID": "Corner ID"}
        ),
        on="Corner ID",
        how="left",
    )

    corners_play_quality = corners_play_quality.sort_values(
        by="Play quality", ascending=False
    )
    corners_play_quality.to_csv(
        f"{OUTPUT_DIR}/corners_play_quality_ranking.csv", index=False
    )

    run_clustering(corners, players)

    # Run similarity-based clustering
    run_similarity_clustering(corners, players)

    print("Done.")


if __name__ == "__main__":
    run_analysis()
