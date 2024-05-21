# tracktour

[![License](https://img.shields.io/pypi/l/tracktour.svg?color=green)](https://github.com/DragaDoncila/tracktour/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tracktour.svg?color=green)](https://pypi.org/project/tracktour)
[![Python Version](https://img.shields.io/pypi/pyversions/tracktour.svg?color=green)](https://python.org)
[![CI](https://github.com/DragaDoncila/tracktour/actions/workflows/ci.yml/badge.svg)](https://github.com/DragaDoncila/tracktour/actions/workflows/ci.yml)

`tracktour` is a simple object tracker based on a network flow linear model. `tracktour` takes a dataframe of detected objects and solves a linear program
(currently using Gurobi, but we will soon add an open source solver interface) to produce tracking results.

`tracktour` is rapidly changing and its API will change without deprecation warnings.

## About `tracktour`

`tracktour` is a purely discrete-optimization-based tracker. It takes the coordinates of detected objects as input, and associates
these objects over time to create complete trajectories, including divisions. Tracktour's only parameter is `k` - the number of
neighbours to consider for possible assignment in the next frame. Using this parameter and very simple distance based cost,
a candidate graph is created, and passed to Gurobi for solving. Once solved, the detected objects and edges that make up the tracks are
returned to the user for inspection.

## Installation

`tracktour` is available as a pip-installable Python package. Running `pip install tracktour` in a virtual environment will install all
required dependencies, but you will need a separate Gurobi Optimizer installation (instructions [here](https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer)).

`tracktour` is tested with all Python versions >=3.8.

**Note** - If you wish to visualize data with `napari` (e.g. as per the Cell Tracking Challenge [example](./examples/build_and_solve_ctc.ipynb)), you will need to separately install it.

## Support

Please feel free to open issues with feature requests, bug reports, questions on usage, etc.

## Usage

The `Tracker` object is the interface for producing tracking solutions. Below is a toy example with explicitly defined detections.

```python
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

See the [toy example](./examples/toy.py) for a complete script, and the [CTC example](./examples/build_and_solve_ctc.ipynb) for visualization in `napari`.

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

**Note:** If using the `ctc` utilities, detections will be extracted for you.

### Cell Tracking Challenge

If you're working with Cell Tracking Challenge formatted datasets, see [the example notebook](./examples/build_and_solve_ctc.ipynb) for producing and visualizing tracks.

You can also use the CLI at the command-line to extract detections, run tracktour, and save output in CTC format.

```sh
# run tracktour with k-neighbours=8
$ tracktour ctc /path/to/seg/ /path/to/save/ -k 8
```

**Note**: Tracktour was recently submitted to the Cell Tracking Challenge. To use the submission version specifically, install `tracktour==0.0.4`.
