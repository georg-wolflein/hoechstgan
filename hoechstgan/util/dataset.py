import typing


def get_channel_files_from_metadata(metadata: dict, **kwargs) -> typing.Iterator[str]:
    for img in metadata["images"]:
        if all(img.get(k, None) == v for (k, v) in kwargs.items()):
            yield img["file"]


def get_channel_file_from_metadata(metadata: dict, channel: str = None, mode: str = None, **kwargs) -> str:
    kwargs = {"channel": channel,
              "mode": mode, **kwargs}
    kwargs = {k: v
              for (k, v) in kwargs.items()
              if k not in ("channel", "mode") or kwargs[k] is not None}
    files = list(get_channel_files_from_metadata(metadata, **kwargs))
    if len(files) != 1:
        raise Exception(
            f"Found {len(files)} instead of 1 unique file matching {', '.join(f'{k}={v}' for (k, v) in kwargs.items())} in metadata ({metadata})")
    file, = files
    return file
