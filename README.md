# Analysis of Arsenal's corners so far in the WSL in 2025/26

This data was collected by hand, watching games on YouTube. See later cells for a definition of the target location.

# Clustering corners using player movement data

How do we know the four groups are correct? Is there a difference between left and right plays or are they just mirrored? Are there more subtle groupings between plays that we can infer from the data (instead of watching video for every corner)? We will try to use clustering to achieve this.

We will assume that the "mop up" role doesn't significantly affect the clustering. We will also assume the "pass target" role is functionally very similar to the "shot target" role, so we will replace it. 

We assume the corners can be mirrored, so we convert all right sided corners to left sided coordinates. The line of symmetry is x=40.


We need to convert into a corner-oriented table, i.e. one row per corner. We will end up with a wide table where there will be a set of columns for each player involved. 

We will plot each of the centroids of the clusters to understand what play they represent


# Play quality

We define a custom "play quality" metric to assess the quality of the play being run at the corner. It is the sum across all players involved at the corner of contributing to a shot, assuming the delivery is perfect. 

Now we have an objective measure of corner quality (excluding the delivery).

I would be wary of making any conclusions based on this small sample but one could imagine using this as a opposition scouting tool to identify the most important players at corners, as this metric accounts for their role and their ability to get into a favourable position.