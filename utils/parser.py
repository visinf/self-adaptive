import argparse
import os

def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=os.path.join(os.getcwd(), "datasets", "gta"))
    parser.add_argument("--checkpoints-root", type=str, default=os.path.join(os.getcwd(), "checkpoints", "runs"))
    parser.add_argument("--num-classes", type=int, default=19, choices=[19, 16], help="Set 19 for a GTA trained model and 16 for a SYNTHIA trained model")
    parser.add_argument("--backbone-name", type=str, default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--arch-type", type=str, default="deeplab", choices=["deeplab", "deeplabv3plus", "hrnet18", "hrnet48"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--dropout", action="store_true", help="Enable dropout during training/Use pre-trained Dropout model")
    parser.add_argument("--alpha", type=float, default=None, help="Between 0.0 and 1.0 for val; For inference: Only set this alpha to [0.0, 1.0] if you want to change the alpha from the checkpoint to a custom alpha")  
    parser.add_argument("--base-lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    return parser

def train_parser():
    parser = base_parser()
    parser.add_argument("--val-dataset-root", type=str, default=os.path.join(os.getcwd(), "datasets", "wilddash"))
    parser.add_argument("--validation-start", type=int, default=0)
    parser.add_argument("--validation-step", type=int, default=1)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lr-scheduler", type=str, choices=["constant", "poly"], default="poly")
    parser.add_argument("--crop-size", nargs='+', type=int, default=[512, 512])
    parser.add_argument("--num-alphas", type=int, default=1, help="1: --alpha is chosen for val, >1: creates alpha linspace vector with [0:num-alphas:1] for val")
    return parser.parse_args()

def val_parser():
    parser = base_parser()
    parser.add_argument("--dataset-split", type=str, default="val")
    parser.add_argument("--source", type=str, default="gta", choices=["gta", "synthia"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Name of checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--only-inf", action="store_true")
    parser.add_argument("--scales", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--flips", action="store_true", help="Apply augmentation flip to all images")
    parser.add_argument("--grayscale",action="store_true", help="Apply grayscaling for Self-adaptation")
    parser.add_argument("--calibration", action="store_true", help="Compute calibration during inference")
    parser.add_argument("--resnet-layers", nargs="+", type=int, default=[1, 2], help="1, 2, 3 and/or 4 which will be frozen for Self-adaptation")
    parser.add_argument("--hrnet-layers", nargs="+", type=int, default=[1, 2], help="1, 2 and/or 3 which will be frozen for Self-adaptation")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")
    return parser.parse_args()