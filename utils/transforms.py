import torch, random
import torchvision.transforms.functional as F
import torchvision.transforms as tf
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple, List, Callable

from datasets.labels import convert_ids_to_trainids, convert_trainids_to_ids


class Compose:

    def __init__(self,
                 transforms: List[Callable]):

        self.transforms = transforms

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:

        for transform in self.transforms:
            img, gt = transform(img, gt)

        return img, gt

class ToTensor:

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:

        img = F.to_tensor(np.array(img))
        gt = torch.from_numpy(np.array(gt)).unsqueeze(0)

        return img, gt

class Resize:

    def __init__(self,
                 resize: Tuple[int]):

        self.img_resize = tf.Resize(size=resize,
                                    interpolation=Image.BILINEAR)
        self.gt_resize = tf.Resize(size=resize,
                                   interpolation=Image.NEAREST)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        img = self.img_resize(img)
        gt = self.gt_resize(gt)

        return img, gt

class ImgResize:

    def __init__(self,
                 resize: Tuple[int, int]):
        self.resize = resize
        self.num_pixels = self.resize[0]*self.resize[1]

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.prod(torch.tensor(img.shape[-2:])) > self.num_pixels:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.resize, mode='bilinear').squeeze(0)
        return img, gt

class ImgResizePIL:

    def __init__(self,
                 resize: Tuple[int]):
        self.resize = resize
        self.num_pixels = self.resize[0]*self.resize[1]

    def __call__(self,
                 img: Image) -> Image:
        if img.height*img.width > self.num_pixels:
            img = img.resize((self.resize[1], self.resize[0]), Image.BILINEAR)
        return img

class Normalize:

    def __init__(self,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):

        self.norm = tf.Normalize(mean=mean,
                                 std=std)

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        img = self.norm(img)

        return img, gt


class RandomHFlip:

    def __init__(self,
                 percentage: float = 0.5):

        self.percentage = percentage

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        if random.random() < self.percentage:
            img = F.hflip(img)
            gt = F.hflip(gt)

        return img, gt


class RandomResizedCrop:

    def __init__(self,
                 crop_size: List[int]):

        self.crop = tf.RandomResizedCrop(size=tuple(crop_size))

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        i, j, h, w = self.crop.get_params(img=img,
                                          scale=self.crop.scale,
                                          ratio=self.crop.ratio)
        img = F.resized_crop(img, i, j, h, w, self.crop.size, Image.BILINEAR)
        gt = F.resized_crop(gt, i, j, h, w, self.crop.size, Image.NEAREST)

        return img, gt

class CenterCrop:

    def __init__(self,
                 crop_size: int):

        self.crop = tf.CenterCrop(size=crop_size)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        img = self.crop(img)
        gt = self.crop(gt)

        return img, gt

class IdsToTrainIds:

    def __init__(self,
                 source: str,
                 target: str):

        self.source = source
        self.target = target

        self.ids_to_trainids = convert_ids_to_trainids

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        gt = self.ids_to_trainids(gt, source=self.source, target=self.target)

        return img, gt


class TrainIdsToIds:

    def __init__(self,
                 source: str,
                 target: str):
        self.source = source
        self.target = target

        self.trainids_to_ids = convert_trainids_to_ids

    def __call__(self,
                 img: torch.Tensor,
                 gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gt = self.trainids_to_ids(gt, source=self.source, target=self.target)

        return img, gt

class ColorJitter:

    def __init__(self, percentage: float = 0.5, brightness: float = 0.3,
                 contrast: float = 0.3, saturation: float = 0.3, hue: float = 0.1):

        self.percentage = percentage
        self.jitter = tf.ColorJitter(brightness=brightness,
                                     contrast=contrast,
                                     saturation=saturation,
                                     hue=hue)

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.percentage:
            img = self.jitter(img)
        return img, gt

class MaskGreyscale:

    def __init__(self, percentage: float = 0.1):
        self.percentage = percentage

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.percentage > random.random():
            img = F.to_grayscale(img, num_output_channels=3)
        return img, gt

class RandGaussianBlur:

    def __init__(self, radius: List[float] = [.1, 2.]):
        self.radius = radius

    def __call__(self,
                 img: Image.Image,
                 gt: Image.Image) -> Tuple[Image.Image, Image.Image]:

        radius = random.uniform(self.radius[0], self.radius[1])
        img = img.filter(ImageFilter.GaussianBlur(radius))

        return img, gt
