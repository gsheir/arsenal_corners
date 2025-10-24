This data was collected by hand, watching games on YouTube. 

I use the following corner zones for data collection:

![](./images/corner_zones.png)


We can plot the frequency of where corners are targeted at and where players are starting or ending their runs:

![](./images/left_front_post_delivery_vs_run_heatmap.png)

# Player roles and play diagrams

I defined roles for each player at each corner:

|Role name	|Description|
|-|-|
|Shot target|	A first receiver that is intended to take a shot|
|Pass target|	A first receiver that is intended to pass the ball (e.g. flick on, head back across goal) |
|Second target|	A second receiver that is intended to take a shot |
|Blocker	|A player who is deliberately blocking an opponent from reaching an area |
|Decoy	|A player who isn’t aiming to win the ball, only to occupy a defender |
|Mop up	|A player who is waiting for a breakaway or loose ball away from the congested areas |

Because I have collected the start and end locations of every player, we can plot a play diagram for each corner. 

![](./images/corner_paths_page_1.png)

# Clustering corners using player movement data

Are there more subtle groupings between plays that we can infer from the data (instead of watching video for every corner)? We will try to use clustering to achieve this.

We will assume that the "mop up" role doesn't significantly affect the clustering. 

We assume the corners can be mirrored, so we convert all right sided corners to left sided coordinates. The line of symmetry is x=40.

## K-means and role aggregated k-means

- Simple k-means, with 5 players per corner, with one-hot encoding of player roles. Used for simplified roles (e.g. just labelling the shot target role, with everything else set to other)
- Role aggregated k-means: Per role, find the count, mean and standard deviation of position to use as features. Better for more roles, potentially improves how the model handles multiple instances of the same role

The cluster centroids are shown below, showing the "average" play for that cluster:

![](./images/all_roles_clustered_corner_paths.png)

We can plot the corners in PCA space to try and visualise the clusters:

![](./images/all_roles_corner_clusters_pca.png)

It can be difficult to get the clusters to be exactly right but using this is a good tool to get an initial idea of the data, without having to watch through every video and group them ourselves. 

## Similarity score + spectral clustering

We define a similarity score based on role-weighted distance, role composition etc., then perform spectral clustering on the similarity matrix. More explainable and takes into account relative importances in roles. I kind of like this approach but it still needs a lot of tuning.

![](./images/corner_similarity_clustering.png)



# Play quality

We define a custom "play quality" metric to assess the quality of the play being run at the corner. It is the sum across all players involved at the corner of contributing to a shot, assuming the delivery is perfect. Each role has a value between 0 and 1, which is multiplied by a score for how they are marked (also between 0 and 1):

```py
ROLES = {
    "Shot target": 1,
    "Pass target": 0.5,
    "Second target": 0.5,
    "Blocker": 0.1,
    "Decoy": 0.1,
    "Mop up": 0.05,
}

MARKING = {
    "Free": 1,
    "Freed by blocker": 1,
    "Blocking": 1,
    "Gained separation": 0.8,
    "Goal marked": 0.3,
    "Watched zonally": 0.3,
    "Front marked": 0.1,
}
```

Now we have an objective measure of corner quality (excluding the delivery).

```
Corner Play Quality:
   Corner ID  Play quality
0     COR-11         2.945
16    COR-36         2.630
1     COR-12         2.225
10    COR-25         2.195
```

I would be wary of making any conclusions based on this small sample but one could imagine using this as a opposition scouting tool to identify the most important players at corners, as this metric accounts for their role and their ability to get into a favourable position.

```
Mean Player Play Quality:
           Player name  Mean play quality
14  Stina Blackstenius           0.432000
0        Alessia Russo           0.419200
13        Steph Catley           0.262609
3          Chloe Kelly           0.260000
10    Lotte Wubben Moy           0.250000
5         Frida Maanum           0.236071
7           Katie Reid           0.232500
2        Caitlin Foord           0.218438
```

# All play diagram plots
![](./images/corner_paths_page_1.png)
![](./images/corner_paths_page_2.png)
![](./images/corner_paths_page_3.png)

# Hand clustering plots
![](./images/hand_clustered_corner_paths_front_post_multiple_shot_targets.png_page_1.png)
![](./images/hand_clustered_corner_paths_front_post_short_lead.png_page_1.png)
![](./images/hand_clustered_corner_paths_front_post_single_shot_target.png_page_1.png)
![](./images/hand_clustered_corner_paths_back_post_with_decoy_to_front.png_page_1.png)
![](./images/hand_clustered_corner_paths_back_post_short_lead.png_page_1.png)

# All clustering plots

## Clustering with all roles aware
![](./images/all_roles_clustered_corner_paths.png)
![](./images/all_roles_clustered_corners_cluster_0.png)
![](./images/all_roles_clustered_corners_cluster_1.png)
![](./images/all_roles_clustered_corners_cluster_2.png)
![](./images/all_roles_clustered_corners_cluster_3.png)
![](./images/all_roles_clustered_corners_cluster_4.png)
![](./images/all_roles_corner_clusters_pca.png)

## Clustering with just shot target (pass targets included)
![](./images/shot_target_aware_clustered_corner_paths.png)
![](./images/shot_target_aware_clustered_corners_cluster_0.png)
![](./images/shot_target_aware_clustered_corners_cluster_1.png)
![](./images/shot_target_aware_clustered_corners_cluster_2.png)
![](./images/shot_target_aware_clustered_corners_cluster_3.png)
![](./images/shot_target_corner_clusters_pca.png)

## Clustering with no roles
![](./images/no_roles_clustered_corner_paths.png)
![](./images/no_roles_clustered_corners_cluster_0.png)
![](./images/no_roles_clustered_corners_cluster_1.png)
![](./images/no_roles_clustered_corners_cluster_2.png)
![](./images/no_roles_corner_clusters_pca.png)

# Similarity + spectral clustering results

```
CLUSTER 0 (4 corners)
----------------------------------------
Corner IDs: COR-2, COR-9, COR-11, COR-31

Key similarities:
  • Shot target players (7 total):
    - Start positions: (41.4, 109.3) - highly consistent
    - End positions: (36.9, 115.1)
  • Pass target players (4 total):
    - Start positions: (39.2, 115.5) - highly consistent
    - End positions: (36.7, 117.8)

CLUSTER 1 (5 corners)
----------------------------------------
Corner IDs: COR-3, COR-6, COR-15, COR-37, COR-46

Key similarities:
  • Shot target players (10 total):
    - Start positions: (40.7, 111.0) - highly consistent
    - End positions: (37.7, 115.2)
  • Second target players (6 total):
    - Start positions: (44.3, 110.0) - highly consistent
    - End positions: (42.2, 114.5)

CLUSTER 2 (3 corners)
----------------------------------------
Corner IDs: COR-8, COR-19, COR-34

Key similarities:
  • Pass target players (4 total):
    - Start positions: (43.3, 113.2) - highly consistent
    - End positions: (40.8, 114.8)
  • Second target players (3 total):
    - Start positions: (42.8, 112.5) - highly consistent
    - End positions: (42.8, 115.5)

CLUSTER 3 (6 corners)
----------------------------------------
Corner IDs: COR-12, COR-13, COR-14, COR-23, COR-24, COR-43

Key similarities:
  • Shot target players (15 total):
    - Start positions: (41.7, 111.1) - highly consistent
    - End positions: (39.7, 114.5)
  • Second target players (1 total):
    - Start positions: (40.0, 105.0) - variable
    - End positions: (38.3, 109.5)

CLUSTER 4 (4 corners)
----------------------------------------
Corner IDs: COR-4, COR-7, COR-20, COR-33

Key similarities:
  • Shot target players (4 total):
    - Start positions: (43.8, 109.9) - highly consistent
    - End positions: (39.2, 113.2)
  • Pass target players (4 total):
    - Start positions: (40.4, 112.9) - moderately consistent
    - End positions: (40.0, 116.2)
  • Second target players (4 total):
    - Start positions: (42.1, 111.4) - highly consistent
    - End positions: (39.2, 114.8)

CLUSTER 5 (2 corners)
----------------------------------------
Corner IDs: COR-35, COR-41

Key similarities:
  • Shot target players (2 total):
    - Start positions: (40.8, 107.2) - highly consistent
    - End positions: (40.0, 114.0)
  • Second target players (7 total):
    - Start positions: (44.3, 111.0) - highly consistent
    - End positions: (41.2, 114.2)
```