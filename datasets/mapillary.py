import os
from PIL import Image
from typing import Callable, Optional, Tuple, List
import torch

class MapillaryDataset(object):
    """
    The Mapillary dataset is required to have following folder structure:
    mapillary/
              training/
                       v1.2/labels/*.png
                       images/*.jpg
    """
    def __init__(self,
                 root,
                 split,
                 transforms: Optional[Callable] = None):

        super(MapillaryDataset, self).__init__()

        self.mode = 'gtFine'
        # Use only subset of 2000 the training images for val, as inference otherwise takes too long
        self.num_images = 2000
        self.images = []
        self.targets = []
        self.transforms = transforms

        val_root = os.path.join(root, "training")
        labels_root = os.path.join(val_root, "v1.2", "labels")
        imgs_root = os.path.join(val_root, "images")
        img_names = os.listdir(imgs_root)
        for i, img_name in enumerate(img_names):
            label_name = img_name.replace(".jpg", ".png")
            img_path = os.path.join(imgs_root, img_name)
            label_path = os.path.join(labels_root, label_name)
            if i < self.num_images:
                self.images.append(img_path)
                self.targets.append(label_path)
            else:
                break

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


def mapillary(root: str,
        split: str,
        transforms: List[Callable]):
    return MapillaryDataset(root=root,
                            split=split,
                            transforms=transforms)