import numpy as np
import pytest
from scipy.spatial import KDTree

from tracktour import Tracker


@pytest.fixture
def get_detections():
    def _get_detections(n_frames=10, n_cells_per_frame=10):
        import numpy as np
        import pandas as pd

        np.random.seed(0)
        detections = pd.DataFrame(
            {
                "t": np.repeat(np.arange(n_cells_per_frame), n_frames),
                "y": np.random.random(n_cells_per_frame * n_frames),
                "x": np.random.random(n_cells_per_frame * n_frames),
            }
        )
        return detections

    return _get_detections


def test_build_trees(get_detections):
    detections = get_detections()
    tracker = Tracker((2, 2))
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
