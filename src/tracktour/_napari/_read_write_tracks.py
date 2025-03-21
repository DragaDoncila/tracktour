import json
import os
import warnings

import pandas as pd


def get_filepath_with_ending(path, ending):
    dir_name = os.path.dirname(path)
    filename = os.path.basename(path).split(".")[0]
    new_filename = f"{filename}.{ending}"
    new_path = os.path.join(dir_name, new_filename)
    return new_path


def get_reader(path):
    """Get reader for paths"""
    if isinstance(path, list):
        pth = path[0]
    else:
        pth = path
    if not (pth.endswith(".csv") or pth.endswith(".json")):
        return None
    if pth.endswith(".json"):
        # can't read json files alone, need the csv next to it
        csv_path = get_filepath_with_ending(pth, "csv")
        if not os.path.exists(csv_path):
            warnings.warn(
                f"Tracktour looked for tracklet file at {csv_path} but couldn't find it. Tracktour can't read the json file alone."
            )
            return None
    return reader


def get_tracklets_array(path):
    df = pd.read_csv(path)
    if "track-id" not in df.columns:
        return
    if "t" not in df.columns:
        return
    if "y" not in df.columns:
        return
    if "x" not in df.columns:
        return
    columns = ["track-id", "t"]
    if "z" in df.columns:
        columns.append("z")
    columns.extend(["y", "x"])
    tracks_only = df[columns]
    tracks_array = tracks_only.to_numpy()
    return tracks_array


def get_graph(path):
    with open(path, "r") as f:
        graph = json.load(
            f, object_hook=lambda d: {int(k): [int(i) for i in v] for k, v in d.items()}
        )

    return graph


def reader(path):
    layer_tuples = []
    null_sentinel = [(None,)]
    if path.endswith(".json"):
        graph_path = path
        tracklet_path = get_filepath_with_ending(path, "csv")
    else:
        tracklet_path = path
        graph_path = get_filepath_with_ending(path, "json")

    tracks = get_tracklets_array(tracklet_path)
    if tracks is None:
        warnings.warn(f"Tracklets not found for {path}. Tracktour can't read tracks.")
        return null_sentinel
    filename = os.path.basename(path).split(".")[0]
    graph = {}
    if not os.path.exists(graph_path):
        warnings.warn(
            f"Graph file {graph_path} not found. Only returning tracklets for {path}."
        )
    else:
        graph = get_graph(graph_path)
    layer_tuples.append((tracks, {"graph": graph, "name": filename}, "tracks"))
    return layer_tuples
