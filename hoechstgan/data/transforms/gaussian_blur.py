from scipy import ndimage
import numpy as np
from PIL import Image

from .transforms import BaseTransform


class GaussianBlur(BaseTransform):

    def forward(self, img):
        sigma = self.opt.get("sigma", 5)
        img = ndimage.gaussian_filter(img, sigma=sigma)
        if img.max() != 0.:
            img = img / img.max() * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img


__export__ = GaussianBlur
