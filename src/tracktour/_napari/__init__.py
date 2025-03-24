try:
    import napari
except ModuleNotFoundError as e:
    raise RuntimeError(
        "napari not found. Cannot use plugin functionality without it. Did you install `tracktour[napari]`?"
    ) from e
