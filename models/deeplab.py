import torch
from typing import Dict

from utils.dropout import add_dropout
from utils.self_adapt_norm import replace_batchnorm
import models.backbone

class DeepLab(torch.nn.Module):

    def __init__(self,
                 backbone_name: str,
                 num_classes: int = 19,
                 dropout: bool = False,
                 alpha: float = 0.0,
                 update_source_bn: bool = True):
        super(DeepLab, self).__init__()
        self.backbone = models.backbone.__dict__[backbone_name](pretrained=True)

        # Initialize classification head
        self.cls_head = torch.nn.Conv2d(
            self.backbone.out_channels, num_classes, kernel_size=1, stride=1, padding=0
        )
        torch.nn.init.normal_(self.cls_head.weight.data, mean=0, std=0.01)
        torch.nn.init.constant_(self.cls_head.bias.data, 0.0)

        # Variable image size during forward pass
        self.img_size = None

        # Add dropout layers after relu
        if dropout:
            add_dropout(model=self)

        # Replace BN layers with SaN layers
        replace_batchnorm(self, alpha=alpha, update_source_bn=update_source_bn)


    def forward(self,
                img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: Batch of input images
        Returns:
            output:
                'backbone': Output features of backbone
                'pred': Segmentation output of images
        """
        # Create output dict of forward pass
        output_dict = {}

        # Set image output size
        self.img_size = img.shape[2:]

        # Compute probabilities for semantic classes at stride 8
        x = self.backbone(img)
        output_dict['backbone'] = x

        # Compute output logits
        x = self._backbone_to_logits(x)
        output_dict['pred'] = x

        return output_dict

    def _backbone_to_logits(self,
                            x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Backbone output features

        Returns:
            x: Upsampled semantic segmentation logits
        """
        # Compute class logits
        x = self.cls_head(x)

        # Bilinear upsampling to full resolution
        x = torch.nn.functional.interpolate(x,
                                            size=self.img_size,
                                            mode='bilinear',
                                            align_corners=True)

        return x

def deeplab(backbone_name: str,
            num_classes: int = 19,
            alpha: float = 0.5,
            update_source_bn: bool = True,
            dropout: bool = False):
    return DeepLab(backbone_name,
                   num_classes,
                   dropout,
                   alpha=alpha,
                   update_source_bn=update_source_bn)