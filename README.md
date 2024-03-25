# tracktour

[![License](https://img.shields.io/pypi/l/tracktour.svg?color=green)](https://github.com/DragaDoncila/tracktour/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tracktour.svg?color=green)](https://pypi.org/project/tracktour)
[![Python Version](https://img.shields.io/pypi/pyversions/tracktour.svg?color=green)](https://python.org)
[![CI](https://github.com/DragaDoncila/tracktour/actions/workflows/ci.yml/badge.svg)](https://github.com/DragaDoncila/tracktour/actions/workflows/ci.yml)

tracktour is a simple object tracker based on a network flow linear model. tracktour takes a dataframe of detected objects and solves a linear program
(currently using Gurobi, but we will soon add an open source solver interface) to produce tracking results.

tracktour is rapidly changing and its API will change without deprecation warnings.

## Usage

The `Tracker` object is the interface for producing tracking solutions. Below is a toy example with explicitly defined detections.

```python
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

# initialize Tracker object
tracker = Tracker(
    im_shape=(100, 100),    # size of the image detections come from. Affects cost of detections appearing/disappearing
    k_neighbours=2          # number of neighbours to consider for assignment in the next frame (default=10)
)
# solve
tracked = tracker.solve(coords)
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

### CLI Tool - Cell Tracking Challenge

If you're working with Cell Tracking Challenge formatted datasets, you can use the CLI to extract detections, run tracktour, and save output in CTC format.

```sh
$ tracktour ctc /path/to/seg/ /path/to/save/ -k 8
```

## Support

Please feel free to open issues with feature requests, bug reports, questions on usage, etc.
