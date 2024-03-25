import os

import typer
from tifffile import imwrite
from typing_extensions import Annotated

from tracktour import Tracker, get_ctc_output, get_im_centers


def _save_results(masks, tracks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_digits = max(len(str(len(masks))), 3)
    for i, frame in enumerate(masks):
        frame_out_name = os.path.join(out_dir, f"mask{str(i).zfill(n_digits)}.tif")
        imwrite(frame_out_name, frame, compression="zlib")

    tracks.to_csv(
        os.path.join(out_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


app = typer.Typer(
    help="Run tracktour on a segmentation and save results in CTC format."
)


@app.command()
def ctc(
    seg_directory: Annotated[
        str,
        typer.Argument(
            help="Input directory containing tiff segmentation for each frame."
        ),
    ],
    out_directory: Annotated[
        str,
        typer.Argument(
            help="Output directory to save tracked segmentation and track info."
        ),
    ],
    k_neighbours: Annotated[
        int,
        typer.Option(
            "--k-neighbours",
            "-k",
            help="Number of neighbours to consider for assignment in next frame, by default 10.",
        ),
    ] = 10,
):
    """Run tracktour on Cell Tracking Challenge formatted data.

    Saves data in Cell Tracking Challenge format.
    """
    ims, detections, _, _, _ = get_im_centers(seg_directory)
    frame_shape = ims.shape[1:]
    location_keys = ("y", "x")
    if len(frame_shape) == 3:
        location_keys = ("z",) + ("y", "x")
    frame_key = "t"
    value_key = "label"

    tracker = Tracker(frame_shape, k_neighbours)
    tracked = tracker.solve(
        detections,
        frame_key=frame_key,
        location_keys=location_keys,
        value_key=value_key,
    )
    sol_graph = tracked.as_nx_digraph()
    relabelled_seg, track_df = get_ctc_output(
        ims, sol_graph, frame_key, value_key, location_keys
    )
    _save_results(relabelled_seg, track_df, out_directory)


# needed because we have just one command for now
@app.callback()
def callback():
    pass


def main():
    app()
