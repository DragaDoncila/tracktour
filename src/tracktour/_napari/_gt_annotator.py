import os

import numpy as np
import pandas as pd
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    PushButton,
    Table,
    create_widget,
)
from napari.utils.notifications import show_error, show_warning
from qtpy.QtWidgets import QSizePolicy

from .._io_util import _get_track_df_from_seg
from ..cli import _save_results


class GroundTruthAnnotator(Container):
    def __init__(
        self,
        layout: str = "vertical",
        scrollable: bool = False,
        labels: bool = True,
    ) -> None:
        super().__init__(layout=layout, scrollable=scrollable, labels=labels)
        self._current_labels_layer = None
        self._graph = dict()
        self._editing_from_combo = True
        self._current_table_data = None

        self._labels_selector = create_widget(
            annotation="napari.layers.Labels", label="Ground Truth Layer"
        )
        self._labels_selector.changed.connect(self._setup_labels_layer)

        self._child_label_selector = ComboBox(choices=[], label="Child Label ID")
        self._parent_label_selector = ComboBox(choices=[], label="Parent Label ID")

        self._add_to_graph_btn = PushButton(label="Add Link")
        self._add_to_graph_btn.clicked.connect(self._add_link_to_graph_from_combo)

        self._parent_table = Table(columns=["Child Track ID", "Parent Track ID"])
        self._parent_table.changed.connect(self._edit_graph_from_table)

        self._open_table = FileEdit(mode="r", label="Existing Graph", filter="*.txt")
        self._open_table_btn = PushButton(label="Load Graph")
        self._open_table_btn.clicked.connect(self._load_existing_graph)

        self._choose_folder = FileEdit(mode="d", label="Target Directory")

        self._save_as_ctc_btn = PushButton(label="Save to CTC Format")
        self._save_as_ctc_btn.clicked.connect(self._save_to_ctc)

        self.extend(
            [
                self._labels_selector,
                self._child_label_selector,
                self._parent_label_selector,
                self._add_to_graph_btn,
                self._parent_table,
                self._open_table,
                self._open_table_btn,
                self._choose_folder,
                self._save_as_ctc_btn,
            ]
        )

        self._setup_labels_layer()

    def _setup_labels_layer(self, label_selector=None):
        if (
            self._labels_selector.value is not self._current_labels_layer
            and self._current_labels_layer is not None
        ):
            self._current_labels_layer.events.selected_label.disconnect(
                self._reset_label_id_choices
            )
        self._current_labels_layer = self._labels_selector.value
        if self._current_labels_layer is not None:
            if not isinstance(self._current_labels_layer.data, np.ndarray):
                show_warning("Can only annotate numpy arrays. Converting now.")
                self._current_labels_layer.data = np.asarray(
                    self._current_labels_layer.data,
                    dtype=self._current_labels_layer.data.dtype,
                )
            self._current_labels_layer.events.selected_label.connect(
                self._reset_label_id_choices
            )
            self._reset_label_id_choices()

    def _setup_parents_table(self):
        header_strs = ["Child Track ID", "Parent Track ID"]

        self._parent_table.setColumnCount(2)
        self._parent_table.setMinimumHeight(40)
        self._parent_table.setHorizontalHeaderLabels(header_strs)
        self._parent_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _get_unique_labels(self):
        if self._current_labels_layer is None:
            return set()
        unique = set(np.unique(self._current_labels_layer.data))
        unique.remove(0)
        return unique

    def _reset_label_id_choices(self, event=None):
        newly_selected_label = self._current_labels_layer.selected_label
        label_id_options = self._get_unique_labels()
        label_id_options.add(newly_selected_label)
        sorted_choices = sorted(list(label_id_options))
        self._child_label_selector.choices = sorted_choices
        self._parent_label_selector.choices = sorted_choices
        self._child_label_selector.value = newly_selected_label
        self._parent_label_selector.value = max(1, newly_selected_label - 1)

    def _get_track_ends(self, track_id):
        track_mask = self._current_labels_layer.data == track_id
        frames_with_label = np.nonzero(np.sum(track_mask, axis=1))
        # the zeroth element of frames_with_label gives us the
        # time coords that contain the mask
        if len(frames_with_label[0] > 0):
            track_start = np.min(frames_with_label[0])
            track_end = np.max(frames_with_label[0])
            return track_start, track_end
        else:
            show_error(f"Parent track with ID {track_id} is not present in any frames.")
            return None, None

    def _add_link_to_graph_from_combo(self, event=None):
        child_label, parent_label = self._get_changed_child_parent_labels_combo(event)
        if not self._check_valid_parent_child_link(child_label, parent_label):
            return
        if child_label in self._graph and self._graph[child_label] == parent_label:
            # we've already handled this change, don't need to do anything else
            return
        self._graph[child_label] = [parent_label]
        self._show_graph_in_table()

    def _check_valid_parent_child_link(
        self, child_label, parent_label, warn_on_override=True
    ):
        if (
            warn_on_override
            and child_label in self._graph
            and self._graph[child_label][0] != parent_label
        ):
            show_warning(
                f"Track {child_label} already has parent {self._graph[child_label][0]}. Overriding..."
            )
        start_of_parent, end_of_parent = self._get_track_ends(parent_label)
        start_of_child, end_of_child = self._get_track_ends(child_label)
        if start_of_child is not None and end_of_parent is not None:
            frame_diff = start_of_child - end_of_parent
            if frame_diff < 1:
                show_error(
                    f"Cannot connect parent {parent_label} ending on frame {end_of_parent} to child {child_label} starting on frame {start_of_child}"
                )
                return False
            if frame_diff > 1:
                show_warning(
                    f"Child track {child_label} starts {frame_diff} frames after parent track {parent_label} ends."
                )
                return True
            return True
        return False

    def _get_changed_child_parent_labels_combo(self, event):
        child_label = self._child_label_selector.value
        parent_label = self._parent_label_selector.value
        return int(child_label), int(parent_label)

    def _get_changed_child_parent_labels(self, changed_dict):
        # we're coming from a table event, grab the info from the event itself
        # not the combo boxes
        changed_label = changed_dict["column_header"]
        other_column = 1 if changed_label == "Child Track ID" else 0
        if changed_label == "Child Track ID" or changed_label == "0":
            child_label = changed_dict["data"]
            parent_label = self._parent_table.data[changed_dict["row"], other_column]
        else:
            child_label = self._parent_table.data[changed_dict["row"], other_column]
            parent_label = changed_dict["data"]
        return child_label, parent_label, changed_label == "Child Track ID"

    def _show_graph_in_table(self):
        dict_as_lists = {
            "Child Track ID": list(self._graph.keys()),
            "Parent Track ID": [parents[0] for parents in self._graph.values()],
        }
        # toggle flag so we know if table edit is coming from GUI or not
        self._editing_from_combo = True
        self._parent_table.set_value(dict_as_lists)
        self._current_table_data = self._parent_table.to_dict(orient="index")
        self._editing_from_combo = False

    def _edit_graph_from_table(self, changed_dict):
        if self._editing_from_combo:
            return
        (
            child_label,
            parent_label,
            child_changed,
        ) = self._get_changed_child_parent_labels(changed_dict)
        if parent_label is None:
            return
        if child_label is None:
            return
        parent_label = int(parent_label)
        child_label = int(child_label)
        original_changed_row = self._current_table_data[changed_dict["row"]]
        original_child_id = original_changed_row["Child Track ID"]
        original_parent_id = original_changed_row["Parent Track ID"]
        row_index = changed_dict["row"]
        if not self._check_valid_parent_child_link(
            child_label, parent_label, warn_on_override=False
        ):
            with self._parent_table.changed.blocked():
                self._parent_table.set_value(
                    pd.DataFrame.from_dict(self._current_table_data, orient="index")
                )
            return
        # if you're editing the parent label, we basically don't care what you do
        # because a parent can have as many children as they want, so we just
        # assign into graph
        if not child_changed:
            self._graph[child_label] = [parent_label]
        else:
            # if you're editing a child label, either you're entering a child value
            # that's already in the graph, or you're entering a new value
            # if it's a value that's already elsewhere in the graph, we remove from the graph and the table
            # if it's a new value that's not in the graph, we should add it, and pop the original child-parent
            # label of this row from the graph
            if self._current_table_data is not None:
                if child_label in self._graph:
                    show_warning(
                        f"Child track {child_label} is already in the table at index {row_index}. Removing row!"
                    )
                    self._parent_table.delete_row(index=row_index)
                self._graph.pop(original_child_id)
                self._graph[child_label] = [parent_label]
        self._current_table_data = self._parent_table.to_dict(orient="index")

    def _save_to_ctc(self, event=None):
        seg = self._current_labels_layer.data
        track_info = _get_track_df_from_seg(seg, self._graph)
        out_dir = self._choose_folder.value
        if out_dir is None or str(out_dir) == os.curdir:
            show_warning(
                f"No directory selected! Saving to current: {os.path.abspath(out_dir)}"
            )
        _save_results(seg, track_info, out_dir)

    def _load_existing_graph(self, event):
        graph_path = self._open_table.value
        if not str(graph_path).endswith(".txt"):
            show_warning("Not a valid graph file path, ignoring.")
            return
        graph = pd.read_csv(graph_path, header=None, sep=" ")
        graph.columns = ["track_id", "start", "end", "parent"]
        self._graph = {
            tid: [parent]
            for tid, parent in zip(graph["track_id"], graph["parent"])
            if parent != 0
        }
        dict_as_lists = {
            "Child Track ID": list(self._graph.keys()),
            "Parent Track ID": [parents[0] for parents in self._graph.values()],
        }
        self._parent_table.set_value(dict_as_lists)
        self._current_table_data = self._parent_table.to_dict(orient="index")
