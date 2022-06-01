import itertools
from pathlib import Path
import typing


def parse_filename(file: typing.Union[Path, str]) -> typing.Dict[str, str]:
    file = Path(file)
    ext = file.suffix[1:]
    name = file.stem[:file.name.index(" [")]
    args = file.stem[file.name.index(" [")+2:file.name.index("].")]
    args = args.split(", ")
    args = {arg.split("=")[0]: arg.split("=")[1] for arg in args}
    return {"name": name, "ext": ext, **args}


def validate_name(name):
    if any(x in str(name) for x in (" [", "].", ", ", "=")):
        raise ValueError(f"Illegal character in input \"{name}\"")


def get_filename(name: str, ext: str, **args) -> str:
    for x in (name, *itertools.chain(*args.items())):
        validate_name(x)
    return f"{name} [{', '.join(f'{k}={v}' for k, v in args.items())}].{ext}"


def find(path: Path, name: str = "*", ext: str = "*", **args) -> typing.Iterator[Path]:
    for file in path.glob(f"{name} [[]*[]].{ext}"):
        parsed_args = parse_filename(file)
        for arg, value in args.items():
            if parsed_args.get(arg, None) != value:
                break
        else:
            yield file
