# tracktour

[![License](https://img.shields.io/pypi/l/tracktour.svg?color=green)](https://github.com/DragaDoncila/tracktour/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tracktour.svg?color=green)](https://pypi.org/project/tracktour)
[![Python Version](https://img.shields.io/pypi/pyversions/tracktour.svg?color=green)](https://python.org)
[![CI](https://github.com/DragaDoncila/tracktour/actions/workflows/ci.yml/badge.svg)](https://github.com/DragaDoncila/tracktour/actions/workflows/ci.yml)

`tracktour` is a simple object tracker based on a network flow linear model. `tracktour` takes a dataframe of detected objects and solves a linear program
(currently using Gurobi, but we will soon add an open source solver interface) to produce tracking results.

`tracktour` is also a `napari` plugin! The plugin allows you to solve, curate and evaluate your solution using three main widgets, as described below.

⚠️ `tracktour` is currently under construction! Its API may change without deprecation warnings.⚠️

## About `tracktour`

`tracktour` is a purely discrete-optimization-based tracker. It takes the coordinates of detected objects as input, and associates
these objects over time to create complete trajectories, including divisions. Tracktour's only parameter is `k` - the number of
neighbours to consider for possible assignment in the next frame. Using this parameter and very simple distance based cost,
a candidate graph is created, and passed to Gurobi for solving. Once solved, the detected objects and edges that make up the tracks are
returned to the user for inspection.

## Installation

`tracktour` is available as a pip-installable Python package. Running `pip install tracktour` in a virtual environment will install all
required dependencies, but you will need a separate Gurobi Optimizer installation (instructions [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer)).

`tracktour` is tested with all Python versions >=3.11.

If you wish to use the `napari` plugin functionality, use `pip install "tracktour[napari]"` if you already have `napari` installed, otherwise `pip install "tracktour[napari]" "napari[all]"`.

## napari Plugin

⚠️More detail coming soon! The plugin is a proof-of-concept only.⚠️

https://github.com/user-attachments/assets/70e10c60-c7d9-4a8a-91af-4518271cf021

`tracktour` is most easily used via its `napari` plugin interface.

The plugin contains three widgets for interacting with your data:

- `Track Solver`: takes a segmentation or points layer and produces your tracking solution.
- `Merge Explorer`: if you allowed your tracks to merge, the Merge Explorer takes you through each merge and allows you to correct them by marking the parents as exiting the frame, or adding new cells that were missed in the segmentation/detection step.
- `Track Annotator`: allows you to pick a sampling strategy and guides you through annotating and correcting ground truth tracks based on your solution. You're shown one edge at a time, with the filled point representing the source, and the hollow point representing the target. You can move either the source or target point around the image to repair the edge. You can also delete either point to signify no incoming/outgoing edge.

After solving, or curating your solution, you can export it to GEFF at any time from each of the widgets. You can then read the solution back in using `napari-geff`.

## Python Usage

The `Tracker` object is the interface for producing tracking solutions. Below is a toy example with explicitly defined detections.

```python
import pandas as pd
from tracktour import Tracker

# define the coordinates of ten detections across three frames.
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
# dataframe with named columns
coords = pd.DataFrame(
    coords,
    columns=["t", "y", "x"]
)

# initialize Tracker object
tracker = Tracker(
    # size of the image detections come from.
    # Affects cost of detections appearing/disappearing
    im_shape=(100, 100),
    # optional segmentation array.
    # can be used for overlap costs (still under construction)
    seg=None,
)
# solve
tracked = tracker.solve(
    coords,
    # number of neighbours to consider for assignment
    # in the next frame (default=10)
    k_neighbours=2,
)
```

The `Tracked` object contains a copy of the detections, potentially reindexed, and a dataframe of edges that make up the solution.
Columns `u` and `v` in `tracked_edges` are direct indices into `tracked_detections`.

```python
print(tracked.tracked_detections)
print(tracked.tracked_edges)
```

You may want to convert the solution into a networkx graph for easier manipulation.

```python
solution_graph = tracked.as_nx_digraph()
```

Or export it to GEFF.

```python
tracked.write_solution_geff("path/to/solution.geff")
```

### Extracting Detections

If you're starting from an image segmentation, you can use the `get_im_centers` or `extract_im_centers` functions.

If your segmentation is already loaded into a numpy array, use `extract_im_centers`. The returned `detections` DataFrame is ready for use with the `Tracker`.

```python
detections, min_t, max_t, corners = extract_im_centers(segmentation)
```

If your segmentation is in Cell Tracking Challenge format and lives in single tiffs per frame in a directory, use `get_im_centers`. This will also return
the segmentation as a numpy array.

```python
seg, detections, min_t, max_t, corners = get_im_centers('path/to/01_RES/')
```

## Support

Please feel free to open issues with feature requests, bug reports, questions on usage, etc.

## Cell Tracking Challenge

**Note**: Tracktour was recently submitted to the Cell Tracking Challenge. To use the submission version specifically, install `tracktour==0.0.4`.
