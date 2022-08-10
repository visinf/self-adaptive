import torchvision
from typing import Any, List, Callable

class CityscapesDataset(torchvision.datasets.Cityscapes):

    def __init__(self,
                 transforms: List[Callable],
                 *args: Any,
                 **kwargs: Any):

        super(CityscapesDataset, self).__init__(*args,
                                                **kwargs,
                                                target_type="semantic")
        self.transforms = transforms


def cityscapes(root: str,
               split: str,
               transforms: List[Callable]):
    return CityscapesDataset(root=root,
                             split=split,
                             transforms=transforms)
