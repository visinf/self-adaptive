# GTA -> Cityscapes (Baseline)
python -W ignore eval.py \
--dataset-root /path/to/cityscapes \
--source gta \
--checkpoints-root checkpoints/runs \
--checkpoint resnet50_gta_alpha_0.0.pth \
--backbone-name resnet50 \
--arch-type deeplab \
--num-classes 19 \
--only-inf

# GTA -> Cityscapes (SaN)
python -W ignore eval.py \
--dataset-root /path/to/cityscapes \
--source gta \
--checkpoints-root checkpoints/runs \
--checkpoint resnet50_gta_alpha_0.1.pth \
--backbone-name resnet50 \
--arch-type deeplab \
--num-classes 19 \
--only-inf

# GTA -> Cityscapes (SaN + TTA)
python -W ignore eval.py \
--dataset-root /path/to/cityscapes \
--source gta \
--checkpoints-root checkpoints/runs \
--checkpoint resnet50_gta_alpha_0.1.pth \
--backbone-name resnet50 \
--num-classes 19 \
--tta \
--flips \
--grayscale \
--batch-size 1 \
--scales 0.25 0.5 0.75

# GTA -> Cityscapes (Self-adaptation)
python -W ignore eval.py \
--dataset-root /path/to/cityscapes \
--source gta \
--checkpoints-root checkpoints/runs \
--checkpoint resnet50_gta_alpha_0.1.pth \
--backbone-name resnet50 \
--num-classes 19 \
--batch-size 1 \
--scales 0.25 0.5 0.75 \
--threshold 0.7 \
--base-lr 0.05 \
--num-epochs 10 \
--flips \
--grayscale \
--weight-decay 0.0 \
--momentum 0.0