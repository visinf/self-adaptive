# GTA
python -W ignore train.py \
--dataset-root /path/to/gta \
--val-dataset-root /path/to/wilddash \
--backbone-name resnet50 \
--arch-type deeplab \
--num-classes 19 \
--batch-size 4 \
--num-epochs 50 \
--crop-size 512 512 \
--validation-start 40 \
--base-lr 5e-3 \
--weight-decay 1e-4 \
--distributed \
--num-alphas 11

# SYNTHIA
python -W ignore train.py \
--dataset-root /path/to/synthia \
--val-dataset-root /path/to/wilddash \
--backbone-name resnet50 \
--arch-type deeplab \
--num-classes 16 \
--batch-size 4 \
--num-epochs 50 \
--crop-size 512 512 \
--validation-start 40 \
--base-lr 5e-3 \
--weight-decay 1e-4 \
--distributed \
--num-alphas 11