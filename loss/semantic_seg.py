import torch
from typing import Dict

class CrossEntropyLoss(torch.nn.Module):

    def __init__(self,
                 ignore_index: int = 255):

        super(CrossEntropyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        self.ignore_index = ignore_index

    def forward(self,
                output: torch.Tensor,
                gt: torch.Tensor):
        """
        Args:
            output: Probabilities for every pixel with stride of 16
            gt: Labeled image at full resolution

        Returns:
            total_loss: Cross entropy loss
        """
        # Compare output and groundtruth at downsampled resolution
        gt = gt.long().squeeze(1)
        loss = self.criterion(output, gt)

        # Compute total loss
        total_loss = (loss[gt != self.ignore_index]).mean()

        return total_loss

class PSPNetLoss(torch.nn.Module):

    def __init__(self,
                 ignore_index: int = 255,
                 alpha: float = 0.0):

        super(PSPNetLoss, self).__init__()
        self.seg_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.cls_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self,
                output_dict: Dict[str, torch.Tensor],
                gt: torch.Tensor):
        """
        Args:
            output_dict: Probabilities for every pixel with stride of 16
            gt: Labeled image at full resolution

        Returns:
            total_loss: Cross entropy loss
        """
        # Compare output and groundtruth at downsampled resolution
        gt = gt.long().squeeze(1)
        seg_loss = self.seg_criterion(output_dict['final'], gt)
        cls_loss = self.cls_criterion(output_dict['aux'], gt)
        total_loss = seg_loss + self.alpha * cls_loss

        return total_loss