import functools
import torch

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def freeze_layers(opts, model: torch.nn.Module):
    if len(opts.resnet_layers) != 0 and "resnet" in opts.backbone_name and "deeplab" in opts.arch_type:
        if opts.arch_type == "deeplab":
            model_name = model.module
        elif opts.arch_type == "deeplabv3plus":
            model_name = model.module.model
        else:
            raise ValueError(f"{opts.arch_type} not compatible with resnet layer freezing")
        for idx, p in enumerate(model_name.named_parameters()):
            if idx <= 3:
                p[1].requires_grad = False
            else:
                break
        for layer in opts.resnet_layers:
            layer = "layer" + str(layer)
            for para in getattr(model_name.backbone, layer).named_parameters():
                para[1].requires_grad = False
    if len(opts.hrnet_layers) != 0 and "hrnet" in opts.arch_type:
        for idx, p in enumerate(model.module.model.named_parameters()):
            if idx <= 3:
                p[1].requires_grad = False
            else:
                break
        for layer_idx in opts.hrnet_layers:
            layer = "transition" + str(layer_idx)
            for para in getattr(model.module.model, layer).named_parameters():
                para[1].requires_grad = False
            layer = "stage" + str(layer_idx + 1)
            for para in getattr(model.module.model, layer).named_parameters():
                para[1].requires_grad = False
