import pandas as pd

from tracktour import Tracker

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

tracker = Tracker(im_shape=(100, 100), k_neighbours=2)
Tracker.DEBUG_MODE = True
# y coordinate is a tenth the magnitude of x coordinate
tracked = tracker.solve(coords, scale=(0.1, 1))
print(tracked.tracked_detections)
print(tracked.tracked_edges)
