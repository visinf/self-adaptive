import torch
import os
from PIL import Image
from typing import Callable, Optional, Tuple, List


class BerkeleyDataset(object):
    """
    First unzip the images: unzip bdd100k_images_10k.zip -d /path/to/bdd100k
    Second unzip the labels in the same directory: unzip bdd100k_sem_seg_labels_trainval.zip -d /path/to/bdd100k
    Third rename the directory from bdd100k to bdd: mv /path/to/bdd100k /path/to/bdd
    The BDD dataset is required to have following folder structure:
    bdd/
        images/
                10k/
                    train/*.jpg
                    test/*.jpg
                    val/*.jpg
        labels/
                sem_seg/
                        masks/
                              train/*.png
                              val/*.png
    """
    def __init__(self,
                 root,
                 split,
                 transforms: Optional[Callable] = None):

        super(BerkeleyDataset, self).__init__()

        self.split = split
        self.transforms = transforms
        images_root = os.path.join(root, "images", "10k", split)
        self.images = []
        targets_root = os.path.join(root, "labels", "sem_seg", "masks", split)
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


def bdd(root: str,
        split: str,
        transforms: List[Callable]):
    return BerkeleyDataset(root=root,
                           split=split,
                           transforms=transforms)
