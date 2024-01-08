import os

import igraph
import pandas as pd

from tracktour import FlowGraph

model_pth = "./_misc"
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
]
coords = pd.DataFrame(coords, columns=["t", "y", "x"])

pixel_vals = [1, 2, 3, 1, 2, 3, 1, 2, 3]
graph = FlowGraph(
    [(0, 0), (100, 100)],
    coords=coords,
    min_t=0,
    max_t=2,
    pixel_vals=pixel_vals,
    migration_only=True,
)
igraph.plot(graph._g, layout=graph._g.layout("rt"))
graph._to_lp(os.path.join(model_pth, "labelled_constraints.lp"))
