import numpy as np
import pandas as pd
import pytest
from scipy.spatial import KDTree

from tracktour import Tracker


@pytest.fixture
def get_detections(n_frames=10, n_cells_per_frame=10):
    def _get_detections():
        np.random.seed(0)
        total_size = n_frames * n_cells_per_frame
        detections = pd.DataFrame(
            {
                "t": np.repeat(np.arange(n_frames), n_cells_per_frame),
                "y": np.random.randint(1, 19, size=total_size),
                "x": np.random.randint(1, 39, size=total_size),
            }
        )
        return detections

    return _get_detections


def test_build_trees(get_detections):
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    assert len(kd_dict) == 10
    for t in range(0, 10):
        assert t in kd_dict
        assert isinstance(kd_dict[t], KDTree)
    # only checking structure of first tree
    one_dict = kd_dict[0]
    assert one_dict.m == 2
    assert one_dict.n == 10
    np.testing.assert_allclose(detections[detections.t == 0][["y", "x"]], one_dict.data)


def test_get_candidate_edges(get_detections):
    detections = get_detections()
    tracker = Tracker(im_shape=(20, 40))
    kd_dict = tracker._build_trees(detections, "t", ("y", "x"))
    edges = tracker._get_candidate_edges(detections, "t", kd_dict)
    # 10 edges for each node, 10 nodes per frame, 9 frames (not last one)
    assert len(edges) == 100 * 9

    # check that each source has 10 edges into next frame
    for u in detections.index.values:
        if detections.iloc[u].t < 9:
            assert len(edges[edges.u == u]) == 10
