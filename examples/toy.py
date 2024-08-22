import pandas as pd

from tracktour import Tracker

# Define 10 detections over 3 frames
coords = [
    (0, 50.0, 50.0),
    (0, 40, 50),
    (0, 30, 57),
    (1, 50, 52),
    (1, 38, 51),
    (1, 29, 60),
    (2, 52, 53),
    (2, 37, 53),
    (2, 28, 64),
    (2, 29, 65),
]
coords = pd.DataFrame(coords, columns=["t", "y", "x"])

# create a Tracker object for this dataset
tracker = Tracker(
    # shape of the image - used to determine distance of detections from border
    im_shape=(100, 100),
    # coordinate scale in each dimension. Particularly important for 3D datasets
    # with a different scale in z compared to x and y.
    scale=(0.5, 0.5),
)
# run tracktour on the coordinates to produce tracks
tracked = tracker.solve(
    # Dataframe of detections
    coords,
    # Column representing the frame number, default 't'
    frame_key="t",
    # Columns (in order) representing the spatial coordinates, default ('y', 'x')
    location_keys=("y", "x"),
    # Number of neighbours in next frame to consider for assignment, default 10
    k_neighbours=2,
)
# the resulting tracked object contains the detections and edges connecting them
print(tracked.tracked_detections)
print(tracked.tracked_edges)
# the tracked object can be converted to a networkx DiGraph for exploration
sol_graph = tracked.as_nx_digraph()
