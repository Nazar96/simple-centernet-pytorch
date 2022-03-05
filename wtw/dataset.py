from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import albumentations as A

from wtw.utils import WTW, load_wtw_annotation, get_cells, get_keypoints, keypoints2cells


class WTWDataset(Dataset):
    def __init__(
            self,
            annotation_dir: str,
            image_dir: str,
            image_size: Optional[Tuple[int, int]] = None,
            use_aug: bool = True,
    ):
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir
        self.tables = os.listdir(image_dir)#[:16]

        h, w = image_size
        self.image_size = (h, w)
        self.output_size = (h//4, w//4)
        
        if use_aug:
            self.transform = self.get_aug_transform()
        else:
            self.transform = self.no_aug_transform()

    def __getitem__(self, item):
        image_name = self.tables[item]
        xml_name = image_name.replace('.jpg', '.xml').replace('.png', '.xml')

        image = cv2.imread(self.image_dir+image_name)
        h, w = image.shape[:2]
        
        annot = load_wtw_annotation(self.annotation_dir+xml_name)
        cells = get_cells(annot)
        kpoints, labels = get_keypoints(cells, h, w)
        
        image, kpoints, labels = self.prep_data(image, kpoints, labels)
        cells = keypoints2cells(kpoints)

        hm, v2c, c2v = WTW()(cells, (512, 512))
        image, hm, v2c, c2v = self.to_tensor([image, hm, v2c, c2v])
        return image, (hm, v2c, c2v)
    
    def __len__(self):
        return len(self.tables)

    @staticmethod
    def to_tensor(elements):
        return [torch.tensor(el).float() for el in elements]

    def prep_data(self, image, keypoints, labels):
        transformed = self.transform(image=image, keypoints=keypoints, class_labels=labels)
        tr_image = transformed["image"]
        tr_keypoints = transformed['keypoints']
        tr_labels = transformed['class_labels']
        tr_image = np.rollaxis(tr_image, -1, 0)/255
        return tr_image, tr_keypoints, tr_labels
    
    @staticmethod
    def get_aug_transform():
        mask_color = (255, 255, 255)
        p = 0.5
        h, w = 512, 512

        aug_list = [
#             A.SafeRotate(5, border_mode=cv2.BORDER_CONSTANT, value=mask_color, p=p),
#             A.Perspective(fit_output=True, pad_val=mask_color, p=p),
        ]

        position = A.PadIfNeeded.PositionType
        pad_aug_list = A.OneOf([
                    A.PadIfNeeded(h//2, w//2, border_mode=cv2.BORDER_CONSTANT, value=mask_color, position=position.BOTTOM_RIGHT),
                    A.PadIfNeeded(h//2, w//2, border_mode=cv2.BORDER_CONSTANT, value=mask_color, position=position.BOTTOM_LEFT),
                    A.PadIfNeeded(h//2, w//2, border_mode=cv2.BORDER_CONSTANT, value=mask_color, position=position.TOP_RIGHT),
                    A.PadIfNeeded(h//2, w//2, border_mode=cv2.BORDER_CONSTANT, value=mask_color, position=position.TOP_RIGHT),
                    A.PadIfNeeded(h//2, w//2, border_mode=cv2.BORDER_CONSTANT, value=mask_color, position=position.CENTER),
        ], p=1)

        transform = A.Compose(aug_list + [
                pad_aug_list,
                A.Resize(h, w),
            ],
            keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=True,
                label_fields=['class_labels']
            ),
        )
        return transform
    
    @staticmethod
    def no_aug_transform():
        mask_color = (255, 255, 255)
        h, w = 512, 512

        position = A.PadIfNeeded.PositionType
        transform = A.Compose([
                A.PadIfNeeded(h//2, w//2, border_mode=cv2.BORDER_CONSTANT, value=mask_color, position=position.CENTER),
                A.Resize(h, w),
            ],
            keypoint_params=A.KeypointParams(
                format='xy',
                remove_invisible=True,
                label_fields=['class_labels']
            ),
        )
        return transform
