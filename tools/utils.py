from tools.settings import ALL_ZONES, MARKING, MIRRORED_CORNER_ZONES, ROLES


def get_start_and_end_counts(players, corner_group):
    corner_group_players = players[players["Corner group"] == corner_group]
    start_counts = (
        corner_group_players.groupby("Start location")["Start location"]
        .count()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False)
    )
    end_counts = (
        corner_group_players.groupby("End location")["End location"]
        .count()
        .reset_index(name="Count")
        .sort_values(by="Count", ascending=False)
    )
    return start_counts, end_counts


def convert_zones_to_xy(player_paths, zone_col, x_col, y_col, all_zones=ALL_ZONES):
    player_paths[x_col] = player_paths[zone_col].map(
        dict(zip(all_zones["zone"], (all_zones["x0"] + all_zones["x1"]) / 2))
    )
    player_paths[y_col] = player_paths[zone_col].map(
        dict(zip(all_zones["zone"], (all_zones["y0"] + all_zones["y1"]) / 2))
    )

    return player_paths


def mirror_right_corners_for_players(
    player_paths, start_x_col="start_x", end_x_col="end_x"
):
    player_paths.loc[player_paths["Side"] == "Right", start_x_col] = (
        80 - player_paths.loc[player_paths["Side"] == "Right", start_x_col]
    )
    player_paths.loc[player_paths["Side"] == "Right", end_x_col] = (
        80 - player_paths.loc[player_paths["Side"] == "Right", end_x_col]
    )

    player_paths.loc[player_paths["Side"] == "Right", "Start location"] = (
        player_paths.loc[player_paths["Side"] == "Right", "Start location"].map(
            MIRRORED_CORNER_ZONES
        )
    )
    player_paths.loc[player_paths["Side"] == "Right", "End location"] = (
        player_paths.loc[player_paths["Side"] == "Right", "End location"].map(
            MIRRORED_CORNER_ZONES
        )
    )
    return player_paths


def add_play_quality_to_players(players, roles=ROLES, marking=MARKING):
    players["Role score"] = players["Role"].map(roles)
    players["Marking score"] = players["Marking"].map(marking)
    players["Play quality"] = players["Role score"] * players["Marking score"]
    return players


def get_mean_play_quality_for_corner_ids(players, corner_ids):
    group_players = players[players["Corner ID"].isin(corner_ids)]
    mean_play_quality = group_players.groupby("Corner ID")["Play quality"].sum().mean()

    return mean_play_quality


def mirror_right_corners_for_corners(corners):
    corners.loc[corners["Side"] == "Right", "Target location"] = corners.loc[
        corners["Side"] == "Right", "Target location"
    ].map(MIRRORED_CORNER_ZONES)

    return corners
