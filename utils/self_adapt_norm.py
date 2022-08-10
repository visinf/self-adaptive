import torch.nn as nn
from copy import deepcopy

from utils.modeling import *


class SelfAdaptiveNormalization(nn.Module):
    def __init__(self,
                 num_features: int,
                 unweighted_stats: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 alpha: float = 0.5,
                 alpha_train: bool = False,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 training: bool = False,
                 update_source: bool = True):
        super(SelfAdaptiveNormalization, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=alpha_train)
        self.alpha_train = alpha_train
        self.training = training
        self.unweighted_stats = unweighted_stats
        self.eps = eps
        self.update_source = update_source
        self.batch_norm = nn.BatchNorm2d(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training and not self.unweighted_stats) or (self.training and self.alpha_train):
            if self.alpha_train:
                self.alpha.requires_grad_()

            # Compute statistics from batch
            x_mean = torch.mean(x, dim=(0, 2, 3))
            x_var = torch.var(x, dim=(0, 2, 3), unbiased=False)

            # Weigh batch statistics with running statistics
            alpha = torch.clamp(self.alpha, 0, 1)
            weighted_mean = (1 - alpha) * self.batch_norm.running_mean.detach() + alpha * x_mean
            weighted_var = (1 - alpha) * self.batch_norm.running_var.detach() + alpha * x_var

            # Update running statistics based on momentum
            if self.update_source and self.training:
                self.batch_norm.running_mean = (1 - self.batch_norm.momentum) * self.batch_norm.running_mean\
                                               + self.batch_norm.momentum * x_mean
                self.batch_norm.running_var = (1 - self.batch_norm.momentum) * self.batch_norm.running_var\
                                              + self.batch_norm.momentum * x_var

            return compute_bn(
                x, weighted_mean, weighted_var,
                self.batch_norm.weight, self.batch_norm.bias, self.eps
                )
        return self.batch_norm(x)

def compute_bn(input: torch.Tensor, weighted_mean: torch.Tensor, weighted_var: torch.Tensor,
               weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    input = (input - weighted_mean[None, :, None, None]) / (torch.sqrt(weighted_var[None, :, None, None] + eps))
    input = input * weight[None, :, None, None] + bias[None, :, None, None]
    return input


def replace_batchnorm(m: torch.nn.Module,
                      alpha: float,
                      update_source_bn: bool = True):
    if alpha is None:
        alpha = 0.0
    for name, child in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            wbn = SelfAdaptiveNormalization(num_features=child.num_features,
                                     alpha=alpha, update_source=update_source_bn)
            setattr(wbn.batch_norm, "running_mean", deepcopy(child.running_mean))
            setattr(wbn.batch_norm, "running_var", deepcopy(child.running_var))
            setattr(wbn.batch_norm, "weight", deepcopy(child.weight))
            setattr(wbn.batch_norm, "bias", deepcopy(child.bias))
            wbn.to(next(m.parameters()).device.type)
            setattr(m, name, wbn)
        else:
            replace_batchnorm(child, alpha=alpha, update_source_bn=update_source_bn)

def reinit_alpha(m: torch.nn.Module,
                 alpha: float,
                 device: torch.device,
                 alpha_train: bool = False):

    layers = [module for module in m.modules() if isinstance(module, SelfAdaptiveNormalization)]
    for i, layer in enumerate(layers):
        layer.alpha = nn.Parameter(torch.tensor(alpha).to(device), requires_grad=alpha_train)
