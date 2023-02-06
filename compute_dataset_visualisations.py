from pylibCZIrw import czi as pyczi
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from tueplots import bundles
from tueplots import figsizes

CHANNELS = ["H3342", "FITC", "Cy3", "Cy5", "AF751"]
WSI_PATH = Path("/data/wsi")
czi_files = list(WSI_PATH.glob("*/*.czi"))
RESULTS_DIR = Path("results")

plt.rcParams.update(bundles.neurips2022())
plt.rcParams.update({"pgf.texsystem": "pdflatex"})


def make_scalar_formatter():
    class ScalarFormatterClass(ScalarFormatter):
        def _set_format(self):
            self.format = "%1.1f"

    formatter = ScalarFormatterClass(useMathText=True)
    formatter.set_powerlimits((0, 0))
    formatter.set_scientific(True)
    return formatter


for channel_index, channel in enumerate(CHANNELS):
    imgs = []
    for czi_file in czi_files:
        with pyczi.open_czi(str(czi_file)) as doc:
            bbox = doc.total_bounding_rectangle
            img = doc.read(roi=bbox,
                           plane={"C": channel_index},
                           zoom=1./64).squeeze()
            imgs.append(img)
    imgs = [img.flatten() for img in imgs]
    imgs = [img[img > 0] for img in imgs]
    sns.displot(imgs, kind="kde", legend=False,
                fill=False, height=2, aspect=2)
    # plt.xlabel(f"Intensity")
    # plt.xlim(0, imgs.max() / 2)
    plt.ylabel("Density")
    plt.gca().yaxis.set_major_formatter(make_scalar_formatter())
    plt.savefig(RESULTS_DIR / f"all_histogram_{channel}.pgf")
    print(f"Saved {channel} histogram")
