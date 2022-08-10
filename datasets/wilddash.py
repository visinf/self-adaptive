import os
import torch
from PIL import Image
from typing import Callable, Optional, Tuple, List

class WilddashDataset(object):
    """
    Unzip the downloaded wd_public_02.zip to /path/to/wilddash
    The wilddash dataset is required to have following folder structure after unzipping:
    wilddash/
            /images/*.jpg
            /labels/*.png
    """
    def __init__(self,
                 root,
                 split,
                 transforms: Optional[Callable] = None):

        super(WilddashDataset, self).__init__()

        self.split = split
        self.transforms = transforms
        images_root = os.path.join(root, "images")
        self.images = []
        targets_root = os.path.join(root, "labels")
        self.targets = []

        for img_name in os.listdir(images_root):
            target_name = img_name.replace(".jpg", ".png")
            self.images.append(os.path.join(images_root, img_name))
            self.targets.append(os.path.join(targets_root, target_name))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


def wilddash(root: str,
             split: str,
             transforms: List[Callable]):
    return WilddashDataset(root=root,
                           split=split,
                           transforms=transforms)