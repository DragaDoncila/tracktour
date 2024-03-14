import argparse

from tracktour import Tracker, get_ctc_output, get_im_centers

parser = argparse.ArgumentParser(
    description="Run tracktour on a segmentation and save results."
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

# generate res_track file

# save masked segmentation

# save res_track file


def main():
    args = parser.parse_args()
    ims, detections, _, _, _ = get_im_centers(args.seg_directory)
    frame_shape = ims.shape[1:]
    location_keys = ("y", "x")
    if len(frame_shape) == 3:
        location_keys = ("z",) + ("y", "x")

    tracker = Tracker(frame_shape, args.k_neighbours)
    tracked = tracker.solve(
        detections, frame_key="t", location_keys=location_keys, value_key="label"
    )
    sol_graph = tracked.as_nx_digraph()

    print(sol_graph.number_of_nodes())

    print(args.out_directory)


if __name__ == "__main__":
    main()
    # main(
    #     '~/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/02_ST/SEG/',
    #     '~/PhD/data/cell_tracking_challenge/ST/Fluo-N2DL-HeLa/02_RES/'
    # )
