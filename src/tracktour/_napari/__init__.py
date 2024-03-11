try:
    import napari
except ModuleNotFoundError as e:
    raise RuntimeError(
        "napari not found. Cannot use plugin functionality without it. Did you install `tracktour[napari]`?"
    ) from e

try:
    import napari_graph
except ModuleNotFoundError as e:
    raise RuntimeError(
        "napari_graph not found. Cannot use plugin functionality without it. Did you install `tracktour[napari]`?"
    ) from e

try:
    import napari_arboretum
except ModuleNotFoundError as e:
    raise RuntimeError(
        "napari_arboretum not found. Cannot use plugin functionality without it. Did you install `tracktour[napari]`?"
    ) from e
