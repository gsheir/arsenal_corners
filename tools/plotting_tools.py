import matplotlib.pyplot as plt
import mplsoccer
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from clustering.role_aggregated_kmeans_clustering import RoleAggregatedKMeansClustering
from tools.settings import OUTPUT_DIR
from tools.utils import get_mean_play_quality_for_corner_ids, get_start_and_end_counts

plt.ioff()


def plot_corner_zones(corner_zones, out_corner_zones, ax=None, out_file=None):
    # Define and plot corner zones
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    pitch = mplsoccer.VerticalPitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="black",
        half=True,
        label=True,
        tick=True,
    )
    pitch.draw(ax=ax)

    for _, row in corner_zones.iterrows():
        ax.add_patch(
            plt.Rectangle(
                (row["x0"], row["y0"]),
                row["x1"] - row["x0"],
                row["y1"] - row["y0"],
                color="red",
                alpha=0.3,
            )
        )
        ax.text(
            (row["x0"] + row["x1"]) / 2,
            (row["y0"] + row["y1"]) / 2,
            row["zone"],
            color="black",
            ha="center",
            va="center",
            fontsize=8,
        )

    for _, row in out_corner_zones.iterrows():
        ax.add_patch(
            plt.Rectangle(
                (row["x0"], row["y0"]),
                row["x1"] - row["x0"],
                row["y1"] - row["y0"],
                color="black",
                alpha=0.3,
            )
        )
        ax.text(
            (row["x0"] + row["x1"]) / 2,
            (row["y0"] + row["y1"]) / 2,
            "out",
            color="black",
            ha="center",
            va="center",
            fontsize=8,
        )

    ax.set_ylim(60, 125)
    ax.set_title("Corner zones", fontsize=14)

    if out_file:
        plt.savefig(out_file)
        plt.close()


def plot_corner_heatmap(
    corners_by_target_location,
    col_name,
    metric_col_name,
    all_zones,
    title="",
    ax=None,
    label=True,
    out_file=None,
):
    """
    Plots the corner target locations on a soccer pitch.

    Parameters:
    - corners_by_target_location: DataFrame with columns 'Target location' and 'Count'.
    - all_zones: DataFrame with columns 'zone', 'x0', 'x1', 'y0', 'y1'.
    """
    # Create a mapping from zone names to counts
    zone_counts = dict(
        zip(
            corners_by_target_location[col_name],
            corners_by_target_location[metric_col_name],
        )
    )

    valid_zones = all_zones["zone"].values

    # Get counts for each valid zone (0 if not present)
    counts = [zone_counts.get(zone, 0) for zone in valid_zones]

    # Create figure
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    else:
        _ = ax.figure

    pitch = mplsoccer.VerticalPitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="black",
        half=True,
        line_zorder=2,
    )
    pitch.draw(ax=ax)

    # Create colormap and normaliser (using 'hot' or 'plasma' which work well on black)
    cmap = plt.cm.Reds
    norm = Normalize(vmin=0, vmax=max(counts) if counts else 1)

    # Plot each zone with color based on count
    for idx, row in all_zones.iterrows():
        zone_name = row["zone"]
        count = zone_counts.get(zone_name, 0)

        # Get color from colormap
        color = cmap(norm(count))

        ax.add_patch(
            plt.Rectangle(
                (row["x0"], row["y0"]),
                row["x1"] - row["x0"],
                row["y1"] - row["y0"],
                color=color,
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
            )
        )

        if label:
            # Add zone label and count
            ax.text(
                (row["x0"] + row["x1"]) / 2,
                (row["y0"] + row["y1"]) / 2,
                f"{count}",
                color="black",
                ha="center",
                va="center",
                fontsize=10,
            )

    ax.set_ylim(80, 125)
    ax.set_title(title, color="black", pad=20, fontsize=14)

    if out_file:
        plt.savefig(out_file)
        plt.close()


def plot_start_end_heatmaps(players, corner_group, all_zones, out_file=None):
    start_counts, end_counts = get_start_and_end_counts(players, corner_group)

    _, ax = plt.subplots(ncols=2, figsize=(8, 4), constrained_layout=True)

    plot_corner_heatmap(
        start_counts,
        col_name="Start location",
        metric_col_name="Count",
        all_zones=all_zones,
        title=f"{corner_group} - Player start locations",
        ax=ax[0],
        label=False,
    )
    plot_corner_heatmap(
        end_counts,
        col_name="End location",
        metric_col_name="Count",
        all_zones=all_zones,
        title=f"{corner_group} - Player end locations",
        ax=ax[1],
        label=False,
    )

    if out_file:
        plt.savefig(out_file)
        plt.close()


role_colours = {
    "Shot target": "red",
    "Pass target": "orange",
    "Second target": "pink",
    "Blocker": "blue",
    "Decoy": "green",
}


def plot_corner_paths(
    corner_paths,
    start_x_col="start_x",
    start_y_col="start_y",
    end_x_col="end_x",
    end_y_col="end_y",
    title="",
    ax=None,
    legend=True,
    role_colours=role_colours,
    out_file=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    pitch = mplsoccer.VerticalPitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="black",
        half=True,
        tick=True,
        label=True,
    )
    pitch.draw(ax=ax)

    for _, row in corner_paths.iterrows():
        ax.arrow(
            row[start_x_col],
            row[start_y_col],
            row[end_x_col] - row[start_x_col],
            row[end_y_col] - row[start_y_col],
            head_width=1,
            head_length=1,
            length_includes_head=True,
            color=role_colours.get(row["Role"], "grey"),
            alpha=0.7,
        )

    ax.set_title(title, fontsize=14)
    if legend:
        ax.legend(
            handles=[
                Patch(color=color, label=role) for role, color in role_colours.items()
            ],
            loc="lower right",
        )

    if out_file:
        plt.savefig(out_file)
        plt.close()


def plot_multiple_corner_paths(
    corner_ids, player_paths, axes=None, out_file_prefix=None, title=""
):
    n_cols = 4
    n_rows = 4
    n_pages = (len(corner_ids) + (n_cols * n_rows) - 1) // (n_cols * n_rows)

    for page in range(n_pages):
        corner_ids_in_page = corner_ids[
            page * n_cols * n_rows : (page + 1) * n_cols * n_rows
        ]

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 4 * n_rows), constrained_layout=True
        )
        axes = axes.flatten()

        for i, corner_id in enumerate(corner_ids_in_page):
            ax = axes[i]
            corner_paths = player_paths[player_paths["Corner ID"] == corner_id]
            plot_corner_paths(
                corner_paths, ax=ax, title=f"Corner ID: {corner_id}", legend=False
            )

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(title)
        if out_file_prefix:
            plt.savefig(f"{out_file_prefix}_page_{page + 1}.png")
            plt.close()


def plot_k_means_results(
    k_means: RoleAggregatedKMeansClustering,
    players,
    out_dir=OUTPUT_DIR,
    filename_prefix="",
):
    # Plot centroid paths using role aggregation reconstruction
    out_file = f"{out_dir}/{filename_prefix}clustered_corner_paths.png"
    fig, ax = plt.subplots(
        nrows=k_means.n_clusters,
        ncols=1,
        figsize=(10, 5 * k_means.n_clusters),
        constrained_layout=True,
    )

    # Ensure ax is always iterable
    if k_means.n_clusters == 1:
        ax = [ax]

    for cluster_id in range(k_means.n_clusters):
        # Get reconstructed paths for this cluster
        all_reconstructed = k_means.reconstructed_paths
        cluster_paths = all_reconstructed[all_reconstructed["Cluster"] == cluster_id]

        # Get the corner IDs that belong to this cluster
        cluster_corners = k_means.clustering_df[
            k_means.clustering_df["Cluster"] == cluster_id
        ]

        plot_corner_paths(
            cluster_paths,
            title=f"Cluster {cluster_id} Centroid",
            ax=ax[cluster_id],
            legend=(cluster_id == 0),  # Only show legend on first plot
        )

        # List corner IDs in this cluster
        corner_ids = cluster_corners.index.tolist()
        ax[cluster_id].text(
            0.5,
            -0.1,
            f"Corner IDs: {corner_ids}",
            transform=ax[cluster_id].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )
        ax[cluster_id].text(
            0.5,
            -0.15,
            f"Mean Play Quality: {get_mean_play_quality_for_corner_ids(players, corner_ids):.3f}",
            transform=ax[cluster_id].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    plt.savefig(out_file)
    plt.close()

    # Plot PCA scatter
    out_file = f"{out_dir}/{filename_prefix}corner_clusters_pca.png"
    plt.figure(figsize=(8, 6))
    for i in range(k_means.n_clusters):
        cluster_data = k_means.clustering_df[k_means.clustering_df["Cluster"] == i]
        cluster_indices = [
            list(k_means.clustering_df.index).index(corner_id)
            for corner_id in cluster_data.index
        ]

        plt.scatter(
            k_means.X_pca[cluster_indices, 0],
            k_means.X_pca[cluster_indices, 1],
            label=f"Cluster {i}",
        )

    # Plot centroids
    plt.scatter(
        k_means.centroids_pca[:, 0],
        k_means.centroids_pca[:, 1],
        s=50,
        c="black",
        marker="X",
        label="Centroids",
    )

    # Add labels for each data point
    for i, corner_id in enumerate(k_means.clustering_df.index):
        plt.text(
            k_means.X_pca[i, 0],
            k_means.X_pca[i, 1],
            str(corner_id),
            fontsize=8,
        )

    plt.title("PCA of Corner Clusters (Role Aggregated)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.savefig(out_file)
    plt.close()

    # Plot individual corners per cluster
    out_file_prefix = "clustered_corners"
    for cluster_id in range(k_means.n_clusters):
        cluster_corners = k_means.clustering_df[
            k_means.clustering_df["Cluster"] == cluster_id
        ]

        if len(cluster_corners) == 0:
            continue

        nrows = int(np.ceil(len(cluster_corners) / 4))
        fig, axs = plt.subplots(
            nrows, 4, figsize=(16, 4 * nrows), constrained_layout=True
        )

        # Handle single row case
        if nrows == 1:
            axs = axs.reshape(1, -1)
        axs = axs.flatten()

        for idx, corner_id in enumerate(cluster_corners.index):
            # Get original player paths for this corner
            corner_player_paths = players[players["Corner ID"] == corner_id]

            if len(corner_player_paths) > 0:
                plot_corner_paths(
                    corner_player_paths,
                    title=f"Corner ID {corner_id}",
                    ax=axs[idx],
                    legend=False,
                )
            else:
                axs[idx].text(
                    0.5,
                    0.5,
                    f"No data for {corner_id}",
                    transform=axs[idx].transAxes,
                    ha="center",
                    va="center",
                )

        # Hide any unused subplots
        for j in range(len(cluster_corners), len(axs)):
            axs[j].axis("off")

        plt.suptitle(f"Cluster {cluster_id} Corners", fontsize=16)
        plt.savefig(
            f"{out_dir}/{filename_prefix}{out_file_prefix}_cluster_{cluster_id}.png"
        )
        plt.close()
