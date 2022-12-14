"""Script to compile the file list of the dataset for BioImage Archive."""

import pandas as pd
import json
from pathlib import Path
import re
import typing
from tqdm import tqdm

SLIDES_DIR = Path("slides")
PATCHES_DIR = Path("patches")
SUBMISSION_DIR = Path("ccrcc")
ICAIRD_NUMBER_PATTERN = re.compile(
    r"ICAIRD(\d+)_MCM2FITC_CD3CY3_CD8CY5_MCK750")

files = []
slides = list(SLIDES_DIR.glob("*.czi"))
patches = list(PATCHES_DIR.glob("**/*.json"))


def icaird_number_from_filename(filename: typing.Union[str, Path]) -> str:
    filename = Path(filename).name
    return ICAIRD_NUMBER_PATTERN.match(filename).group(1)


def add_file(file: Path, **kwargs):
    files.append({
        "Files": str(SUBMISSION_DIR / file),
        **kwargs
    })


for f in tqdm(slides, desc="Processing slides"):
    add_file(f, type="whole slide image",
             case_id=icaird_number_from_filename(f))

for f in tqdm(patches, desc="Processing patches"):
    with f.open() as fp:
        data = json.load(fp)
    patch_meta = dict(patch_x=data["x"],
                      patch_y=data["y"],
                      patch_width=data["w"],
                      patch_height=data["h"])
    add_file(f,
             type="patch metadata",
             case_id=icaird_number_from_filename(f),
             **patch_meta)
    for img in data["images"]:
        add_file(f.with_name(img["file"]),
                 type="patch image",
                 case_id=icaird_number_from_filename(f),
                 **patch_meta,
                 **{f"patch_{k}": v for k, v in img.items() if k not in ("file", "mask_type")})


df = pd.DataFrame(files)
print(df)
df.to_csv("file_list.tsv", index=False, sep="\t", float_format="%.0f")
