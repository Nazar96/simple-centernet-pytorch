from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

from wtw.utils import WTW


class WTWDataset(Dataset):
    def __init__(
            self,
            annotation_dir: str,
            image_dir: str,
            image_size: Optional[Tuple[int, int]] = None
    ):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.tables = os.listdir(image_dir)

        h, w = image_size
        self.image_size = (h, w)
        self.output_size = (h//4, w//4)

    def __getitem__(self, item):
        image_name = self.tables[item]
        xml_name = image_name.replace('.jpg', '.xml')

        image = cv2.imread(self.image_dir+image_name)
        image = self.prep_image(image)

        hm, dm, v2c, c2v = WTW(self.annotation_dir+xml_name)(self.output_size)
        image, hm, dm, v2c, c2v = self.to_tensor([
            image, hm, dm, v2c, c2v
        ])
        return image, (hm, dm, v2c, c2v)
    
    def __len__(self):
        return len(self.tables)

    @staticmethod
    def to_tensor(elements):
        return [torch.tensor(el).float() for el in elements]

    def prep_image(self, image):
        if self.image_size:
            image = cv2.resize(image, self.image_size)
        inp = np.expand_dims(np.rollaxis(image, -1, 0), 0)/255
        return inp
