import torchvision.transforms.functional as F
import torchvision.transforms as tf
from PIL import Image, ImageFilter
import torch
from typing import List, Any
import os
import datasets
from utils import transforms


class TrainTestAugDataset:

    def __init__(self,
                 device,
                 source,
                 crop_size: List[int],
                 transforms_list: transforms.Compose = transforms.Compose([]),
                 only_inf: bool = False,
                 combined_augmentation: bool = True,
                 ignore_index: int = 255,
                 threshold: float = 0.7,
                 getinfo: bool = False,
                 tta: bool = False,
                 flip_all_augs: bool = False,
                 flips: bool = True,
                 scales: list = [1.0],
                 greyscale: bool = False,
                 colorjitter: bool = False,
                 gaussblur: bool = False,
                 rotation: bool = False,
                 rot_angle: int = 30,
                 jitter_factor: float = 0.4,
                 gauss_radius: float = 1.0,
                 *args: Any,
                 **kwargs: Any):

        self.root = kwargs['root']
        self.device = device
        self.source = source
        self.target = os.path.basename(self.root)
        self.dataset = datasets.__dict__[self.target](root=self.root,
                                                      split=kwargs['split'],
                                                      transforms=transforms_list)
        self.combined_augmentation=combined_augmentation
        self.dataset.transforms = transforms_list
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.getinfo = getinfo
        self.tta = tta
        self.scales = scales
        self.flip_all_augs = flip_all_augs
        self.flips = flips
        self.greyscale = greyscale
        self.colorjitter = colorjitter
        self.gaussblur = gaussblur
        self.rotation = rotation
        self.rot_angle = int(rot_angle)
        self.jitter_factor = jitter_factor
        self.gauss_radius = gauss_radius
        self.augs = [None]
        if self.flips: self.augs.append("flip")
        if self.greyscale: self.augs.append("grey")
        if self.colorjitter: self.augs.append("jitter")
        if self.gaussblur: self.augs.append("gauss")
        if self.rotation: self.augs.append("rot")
        self.resize_image_pre = transforms.ImgResizePIL(crop_size)
        self.only_inf = only_inf

    def __getitem__(self, idx: int):

        image, target = self.dataset.__getitem__(idx)
        # Resize image
        image = self.resize_image_pre(image)

        crop_imgs = []
        transforms_list = []
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.IdsToTrainIds(source=self.source, target=self.target),
                                    transforms.Normalize()])

        if self.only_inf:
            image, target = trans(image, target)
            return image, target, [], []

        if self.combined_augmentation: self.augs = [None, None]
        for scale in self.scales:
            for idx, aug in enumerate(self.augs):
                # Apply scaling
                i, j = 0, 0
                w, h = [int(i*scale) for i in image.size]
                crop_img = image.resize((w, h), Image.BILINEAR)

                # Additional augmentations on every duplicate of the scale
                flip_action, rot_action, grey_action, jitter_action, gauss_action = False, False, False, False, False
                if self.flip_all_augs and idx != 0:
                    flip_action = True
                    crop_img = F.hflip(crop_img)

                if self.combined_augmentation and idx == 1:
                    if self.flips:
                        flip_action = True
                        crop_img = F.hflip(crop_img)
                    if self.rotation:
                        rot_action = True
                        crop_img = F.rotate(crop_img, angle=self.rot_angle, expand=True, fill=0)
                    if self.colorjitter:
                        jitter_action = True
                        crop_img = tf.ColorJitter(brightness=self.jitter_factor,
                                                  contrast=self.jitter_factor,
                                                  saturation=self.jitter_factor,
                                                  hue=min(0.1, self.jitter_factor))(crop_img)
                    if self.gaussblur:
                        gauss_action = True
                        crop_img = crop_img.filter(ImageFilter.GaussianBlur(self.gauss_radius))
                    if self.greyscale:
                        grey_action = True
                        crop_img = F.to_grayscale(crop_img, num_output_channels=3)

                if not self.combined_augmentation:
                    if aug == "flip":
                        flip_action = True
                        crop_img = F.hflip(crop_img)
                    if aug == "rot":
                        rot_action = True
                        crop_img = F.rotate(crop_img, angle=self.rot_angle, expand=True, fill=0)
                    if aug == "jitter":
                        jitter_action = True
                        crop_img = tf.ColorJitter(brightness=self.jitter_factor,
                                                  contrast=self.jitter_factor,
                                                  saturation=self.jitter_factor,
                                                  hue=min(0.1, self.jitter_factor))(crop_img)
                    if aug == "gauss":
                        gauss_action = True
                        crop_img = crop_img.filter(ImageFilter.GaussianBlur(self.gauss_radius))
                    if aug == "grey":
                        grey_action = True
                        crop_img = F.to_grayscale(crop_img, num_output_channels=3)

                crop_img, _ = trans(crop_img, crop_img)
                transforms_list.append((i, j, w, h, flip_action, rot_action,
                                        self.rot_angle, grey_action, jitter_action, gauss_action))
                crop_imgs.append(crop_img)

        image, target = trans(image, target)
        return image, target, crop_imgs, transforms_list

    def __len__(self) -> int:
        return len(self.dataset.images)

    def create_pseudo_gt(self,
                         crops_soft: torch.Tensor,
                         crop_transforms: List[List[torch.Tensor]],
                         out_shape: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crops_soft: Tensor with model outputs of crops (N, C, H, W)
            crop_transforms: List with transformations (e.g. random crop and hflip parameters)
            out_shape: Tensor with output shape

        Returns:
            pseudo_gt: Pseudo ground truth based on softmax probabilities
        """

        num_classes = crops_soft[0].shape[1]
        crops_soft_all = torch.ones(len(crops_soft), num_classes, *out_shape[-2:]) * self.ignore_index
        for crop_idx, (crop_soft, crop_transform) in enumerate(zip(crops_soft, crop_transforms)):
            i, j, h, w, flip_action, rot_action, rot_angle, grey_action, jitter_action, gauss_action = crop_transform

            # Reaugment Images
            if rot_action:
                # Rotate back
                crop_soft = F.rotate(crop_soft, angle=-int(rot_angle))
                crop_soft = tf.CenterCrop(size=(h, w))(crop_soft)

            if flip_action:
                crop_soft = F.hflip(crop_soft)

            # Scale to original scale
            crop_soft = torch.nn.functional.interpolate(
                crop_soft, size=[*out_shape[-2:]], mode='bilinear', align_corners=True
            )
            h, w = out_shape[-2:]
            crops_soft_all[crop_idx, :, i:i+h, j:j+w] = crop_soft.squeeze(0)

        pseudo_gt = torch.mean(crops_soft_all, dim=0)

        if self.tta:
            pseudo_gt = pseudo_gt.unsqueeze(0)
        else:
            # Create mask to compare only max predictions
            compare_mask = torch.amax(pseudo_gt, dim=0, keepdim=True) == pseudo_gt
            class_threshold = self.threshold * torch.max(torch.max(pseudo_gt, dim=1)[0], dim=1)[0]
            if self.getinfo: print(f"Class thresholds: {class_threshold}")
            class_threshold = class_threshold.unsqueeze(1).unsqueeze(1).repeat(1, pseudo_gt.shape[1],
                                                                   pseudo_gt.shape[2])
            # Set ignore indices for pixels having not enough pixels or ignore indices
            threshold_mask = class_threshold < pseudo_gt
            threshold_mask = torch.amax(torch.mul(threshold_mask, compare_mask), dim=0)
            final_mask = threshold_mask.unsqueeze(0).unsqueeze(0)

            pseudo_gt = torch.argmax(pseudo_gt, dim=0, keepdim=True).unsqueeze(0)
            pseudo_gt[~final_mask] = self.ignore_index

        return pseudo_gt
