import argparse
import os

from tifffile import imwrite

from tracktour import Tracker, get_ctc_output, get_im_centers


def _save_results(masks, tracks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_digits = len(str(len(masks)))
    for i, frame in enumerate(masks):
        frame_out_name = os.path.join(out_dir, f"mask{str(i).zfill(n_digits)}.tif")
        imwrite(frame_out_name, frame)

    tracks.to_csv(
        os.path.join(out_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


def _run_tracktour(seg_directory, out_directory, k_neighbours=10):
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


parser = argparse.ArgumentParser(
    description="Run tracktour on a segmentation and save results in CTC format."
)
parser.add_argument(
    "seg_directory",
    type=str,
    help="Input directory containing tiff segmentation for each frame.",
)
parser.add_argument(
    "out_directory",
    type=str,
    help="Output directory to save tracked segmentation and track info.",
)
parser.add_argument(
    "-k",
    "--k-neighbours",
    dest="k_neighbours",
    default=10,
    type=int,
    help="Number of neighbours to consider for assignment in next frame, by default 10.",
)


def main():
    args = parser.parse_args()
    _run_tracktour(args.seg_directory, args.out_directory, args.k_neighbours)


if __name__ == "__main__":
    main()
