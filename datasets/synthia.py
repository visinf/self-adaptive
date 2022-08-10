from PIL import Image
from typing import Optional, Callable, Tuple, List
import os
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SynthiaDataset(object):
    """
    The Synthia dataset is required to have following folder structure:
    synthia/
            leftImg8bit/
                        train/seq_id/*.png
                        val/seq_id/*.png
            gtFine/
                train/seq_id/*.png
                val/seq_id/*.png
    """
    def __init__(self,
                 root,
                 split,
                 transforms: Optional[Callable] = None):

        super(SynthiaDataset, self).__init__()

        self.mode = 'gtFine'
        self.images_dir = os.path.join(root, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root, self.mode, split)
        self.split = split
        self.images = []
        self.targets = []
        self.transforms = transforms

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_id = '{}'.format(file_name.split('_leftImg8bit')[0])
                target_suffix = "_gtFine_labelIds"
                target_ext = ".png"
                target_name = target_id + target_suffix + target_ext

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)

def synthia(root: str,
            split: str,
            transforms: List[Callable]):
    return SynthiaDataset(root=root,
                          split=split,
                          transforms=transforms)