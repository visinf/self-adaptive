import os
from typing import Tuple, List, Callable, Optional
from PIL import Image
import torch

class IDDDataset(object):
    """
    Follow these steps to prepare the IDD dataset:
    - Unpack the downloaded dataset: tar -xf idd-segmentation.tar.gz -C /path/to/IDD_Segmentation/
    - Rename the directory from IDD_Segmentation to idd: mv /path/to/IDD_Segmentation /path/to/idd
    Create train IDs from polygon annotations:
    - wget https://github.com/AutoNUE/public-code/archive/refs/heads/master.zip
    - unzip master.zip -d iddscripts
    - export PYTHONPATH="${PYTHONPATH}:iddscripts/public-code-master/helpers/"
    - pip install -r iddscripts/public-code-master/requirements.txt
    - python iddscripts/public-code-master/preperation/createLabels.py --datadir /path/to/idd --id-type csTrainId --num-workers 1
    - rm -rf iddscripts
    The IDD dataset is required to have following folder structure:
    idd/
        leftImg8bit/
                    train/city/*.png
                    test/city/*.png
                    val/city/*.png
        gtFine/
               train/city/*.png
               test/city/*.png
               val/city/*.png
    """
    def __init__(self,
                 root,
                 split,
                 transforms: Optional[Callable] = None):

        super(IDDDataset, self).__init__()

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
                target_name = file_name.split("_leftImg8bit.png")[0] + "_gtFine_labelcsTrainIds.png"
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

def idd(root: str,
        split: str,
        transforms: List[Callable]):
    return IDDDataset(root=root,
                      split=split,
                      transforms=transforms)