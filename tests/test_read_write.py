import json

import numpy as np
import pandas as pd
import pytest

from tracktour._napari._read_write_tracks import (
    get_graph,
    get_reader,
    reader,
    write_tracks,
)


@pytest.fixture
def dummy_tracks():
    tracks_data = [
        [1, 0, 236, 0],
        [1, 1, 236, 100],
        [1, 2, 236, 200],
        [1, 3, 236, 500],
        [1, 4, 236, 1000],
        [2, 0, 436, 0],
        [2, 1, 436, 100],
        [2, 2, 436, 200],
        [2, 3, 436, 500],
        [2, 4, 436, 1000],
        [3, 0, 636, 0],
        [3, 1, 636, 100],
        [3, 2, 636, 200],
        [3, 3, 636, 500],
        [3, 4, 636, 1000],
        [4, 0, 200, 50],
        [5, 1, 204, 54],
        [5, 2, 204, 55],
        [6, 1, 196, 46],
        [6, 2, 196, 47],
    ]
    tracks_df = pd.DataFrame(tracks_data, columns=["track_id", "t", "y", "x"])
    graph = {5: [4], 6: [4]}
    return tracks_df, graph


def save_tracks(tmp_path, tracks_df, graph, use_diff_json=False):
    tracks_df.to_csv(tmp_path / "dummy.csv", index=False)
    json_name = "dummy.json" if not use_diff_json else "dummy_diff.json"
    with open(tmp_path / json_name, "w") as f:
        json.dump(graph, f)


def test_get_reader(tmp_path, dummy_tracks):
    # random extension returns None
    assert get_reader("dummy.txt") is None

    # csv tries to be read
    assert get_reader("dummy.csv") is not None

    # can't read json if csv is missing
    with pytest.warns(match="Tracktour looked for tracklet file"):
        assert get_reader("dummy.json") is None

    # once we write the csv and json, just json is ok
    tracks_df, graph = dummy_tracks
    save_tracks(tmp_path, tracks_df, graph)
    assert get_reader(str(tmp_path / "dummy.json")) is not None


def test_reader_csv_only(tmp_path, dummy_tracks):
    tracks_df, graph = dummy_tracks
    pth = tmp_path / "dummy.csv"
    tracks_df.to_csv(pth, index=False)

    with pytest.warns(match=f"Graph file .*json not found"):
        tracks_tuple = reader(str(pth))
        assert len(tracks_tuple) == 1
        data, meta, lyr_tpe = tracks_tuple[0]
        assert data.shape == (len(tracks_df), 4)
        assert meta["graph"] == {}
        assert meta["name"] == "dummy"


def test_reader_csv_and_json(tmp_path, dummy_tracks):
    tracks_df, graph = dummy_tracks
    save_tracks(tmp_path, tracks_df, graph)

    paths = [tmp_path / "dummy.json", tmp_path / "dummy.csv"]
    for path in paths:
        tracks_tuple = reader(str(path))
        assert len(tracks_tuple) == 1
        data, meta, lyr_tpe = tracks_tuple[0]
        assert data.shape == (len(tracks_df), 4)
        assert meta["graph"] == graph
        assert meta["name"] == "dummy"
        assert lyr_tpe == "tracks"


def test_reader_gives_nothing(tmp_path, dummy_tracks):
    tracks_df, graph = dummy_tracks
    tracks_df.drop(columns=["track_id"], inplace=True)  # incomplete data
    save_tracks(tmp_path, tracks_df, graph)
    with pytest.warns(match="Tracklets not found for"):
        assert reader(str(tmp_path / "dummy.csv")) == [(None,)]


def test_writer_no_graph(tmp_path, dummy_tracks):
    tracks_df, _ = dummy_tracks
    write_tracks(str(tmp_path / "dummy.csv"), tracks_df.values, {"graph": {}})
    assert (tmp_path / "dummy.csv").exists()
    assert not (tmp_path / "dummy.json").exists()


def test_writer_with_graph(tmp_path, dummy_tracks):
    tracks_df, graph = dummy_tracks
    write_tracks(str(tmp_path / "dummy.csv"), tracks_df.values, {"graph": graph})
    assert (tmp_path / "dummy.csv").exists()
    assert (tmp_path / "dummy.json").exists()
    assert get_graph(tmp_path / "dummy.json") == graph
    written_df = pd.read_csv(tmp_path / "dummy.csv")
    np.testing.assert_array_equal(written_df.columns, ["track_id", "t", "y", "x"])
    np.testing.assert_array_equal(written_df.values, tracks_df.values)


# def test_napari_round_trip(tmp_path, dummy_tracks, make_napari_viewer):
#    import napari
#
#     tracks_df, graph = dummy_tracks
#     write_tracks(str(tmp_path / "dummy.csv"), tracks_df.values, {"graph": graph})

#     viewer = make_napari_viewer()
#     viewer.open(str(tmp_path / "dummy.csv"), plugin="tracktour")
#     assert len(viewer.layers) == 1
#     viewer.layers.save(str(tmp_path / "saved_dummy.csv"), plugin="tracktour", selected=True)
#     assert (tmp_path / "saved_dummy.csv").exists()
#     assert (tmp_path / "saved_dummy.json").exists()
