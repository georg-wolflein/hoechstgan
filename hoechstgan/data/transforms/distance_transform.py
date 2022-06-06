from scipy import ndimage
import numpy as np
from PIL import Image

from .transforms import BaseTransform


class DistanceTransform(BaseTransform):

    def forward(self, img):
        img = ndimage.distance_transform_edt(img)
        if img.max() != 0.:
            img = img / img.max() * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img


__export__ = DistanceTransform
