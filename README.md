# Semantic Self-adaptation: Enhancing Generalization with a Single Sample

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the official implementation of our paper:

**Semantic Self-adaptation: Enhancing Generalization with a Single Sample**<br>
[Sherwin Bahmani](https://sherwinbahmani.github.io)\*,
[Oliver Hahn](https://olvrhhn.github.io)\*,
[Eduard Zamfir](https://eduardzamfir.github.io)\*,
[Nikita Araslanov](https://arnike.github.io),
[Daniel Cremers](https://vision.in.tum.de/members/cremers),
and [Stefan Roth](https://www.visinf.tu-darmstadt.de/visual_inference/people_vi/stefan_roth.en.jsp)<br>

*equal contribution, [[arXiv](https://arxiv.org)]


<table>
<tr>
<td><img src="assets/main.gif" width="384" height="384"/></td>
<td>
<i>Self-adaptation</i> adjusts only the inference process, while standard regularization is employed during network training.
Given a single unlabeled test sample as the input, self-adaptation customizes the parameters of convolutional and Batch Normalization layers, before producing the output for that sample.
Self-adaptation significantly improves out-of-distribution generalization and sets new state-of-the-art accuracy on multi-domain benchmarks.
<br><br><br><br><br><br><br><br><br><br>
</td>
</tr>
</table>

## Installation
This project was originally developed with Python 3.8, PyTorch 1.9, and CUDA 11.0. The training with DeepLabv1 ResNet50 requires 
one NVIDIA GeForce RTX 2080 (11GB). For DeepLabv1 ResNet101 and all DeepLabv3+ variants we used a single NVIDIA Tesla V100 (32GB) as these architectures require more memory.

- Create conda environment:
```
conda create --name selfadapt
source activate selfadapt
```
- Install PyTorch >=1.9 (see [PyTorch instructions](https://pytorch.org/get-started/locally/)). For example,
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
- Install the dependencies:
```
pip install -r requirements.txt
```
- Download the datasets ([BDD](https://bdd-data.berkeley.edu/), [Cityscapes](https://www.cityscapes-dataset.com), [GTA](https://download.visinf.tu-darmstadt.de/data/from_games/), [IDD](https://idd.insaan.iiit.ac.in), [Mapillary](https://www.mapillary.com/dataset/vistas?pKey=rwbBtYKofke2NeLIvj8j-A&lat=20&lng=0&z=1.5), [SYNTHIA](http://synthia-dataset.net/downloads/), [WildDash](https://wilddash.cc)).
- Please follow `datasets/{bdd, cityscapes, gta, idd, mapillary, synthia, wilddash}.py` for instructions to setup the datasets.

We use GTA and SYNTHIA as source domains in order to train our models. As for our development set, we decided to use the WildDash dataset for validation during training. 
We evaluate our method on Cityscapes, BDD, IDD, and Mapillary. In the case of GTA as our source domain, we removed the broken GTA image/label pairs from the training set.
We used all SYNTHIA-RAND-CITYSCAPES training images. We evaluate on all validation images of each target domain.

## Training
### Baseline
Starting from an ImageNet initialization, we train the DeepLabv1/DeepLabv3+/HRNet baseline on GTA and SYNTHIA, while we use WildDash as the validation set during training.
To run the training, please use the following command or use scripts/train.sh:
- ```--dataset-root```: Path to dataset folder containing source data. Path must end with name of dataset specified in `train.py`. Example: `/user/data/cityscapes`
- ```--val-dataset-root```: Path to dataset folder containing validation data.
- ```--backbone-name```: Use either ```resnet50``` or ```resnet101``` as backbone.
- ```--arch-type```: Use either ```deeplab```, ```deeplabv3plus``` or ```hrnet18``` as a model.
- ```--num-classes```: ```19``` if source is ```gta```, ```16``` if source is ```synthia```.
- ```--distributed```: Use PyTorch's DistributedDataParallel wrapper for distributed training.
- ```--dropout```: Set true to run a baseline training with Dropout as needed for MC-Dropout.

Hyperparameters for training:
- ```--batch-size```: Number of images per batch (default:```4```).
- ```--num-epochs```: Number of epochs during training (default:```50```).
- ```--crop-size```: Size of crops used for training (default:```512 512```).
- ```--validation-start```: Start validation after a certain number of epochs (default: ```40```).
- ```--base-lr```: Initial learning rate (default: ```5e-3```).
- ```--lr-scheduler```: Choose between ```constant``` LR or ```poly``` LR decay (default: ```poly```).
- ```--weight-decay```: Weight decay (default: ```1e-4```).
- ```--num-alphas```: Creates vector with ```[0:num-alphas:1]``` for validation (default: ```11```).
```
python train.py --dataset-root DATASET-ROOT --val-dataset-root VAL-DATASET-ROOT --backbone-name [resnet50|resnet101] --arch-type [deeplab|deeplabv3plus|hrnet18] --num-classes [19|16] --distributed --batch-size 4 --num-epochs 50 --crop-size 512 512 --validation-start 40 --base-lr 5e-3 --weight-decay 1e-4 --num-alphas 11
```

## Inference
### Standard
To run inference, please use the following command or use scripts/eval.sh:
- ```--source```: Specifies on which source domain the current checkpoint was trained on.
- ```--checkpoint```: Filename of desired checkpoint.
- ```--checkpoints-root```: Path to folder containing checkpoint.
- ```--only-inf```: Standard inference will be performed.
- ```--num-classes```: ```19``` if source is ```gta```, ```16``` if source is ```synthia```.
- ```--mixed-precision```: Use mixed precision for the tensor operations in different layers
```
python eval.py --dataset-root DATASET-ROOT --source [gta|synthia] --checkpoints-root CHECKPOINTS-ROOT --checkpoint CHECKPOINT --backbone-name [resnet50|resnet101] --arch-type [deeplab|deeplabv3plus|hrnet18] --num-classes [19|16] --only-inf
```
### Calibration
```--calibration```: To evaluate the model's calibration, add the flag to the above command.

### TTA
For performing Test-Time Augmentation, replace ```--only-inf``` with ```--tta```. The following arguments define the augmentations made on each single test sample. Those augmented images form the augmented batch together with the initial image.
- ```--batch-size```: Use a single sample from the validation set to generate augmented batch (default:```1```).
- ```--scales```: Defines scaling ratio (default: ```0.25 0.5 0.75```).
- ```--flips```: Add a flipped image for all scales.
- ```--grayscale```: Add a grayscaled image for all scales.
```
python eval.py --dataset-root DATASET-ROOT --source [gta|synthia] --checkpoints-root CHECKPOINT-ROOT --checkpoint CHECKPOINT --backbone-name [resnet50|resnet101] --num-classes [19|16] --tta --flips --grayscale --batch-size 1 --scales 0.25 0.5 0.75 --num-workers 8
```
### Self-adaptation
During self-adaptation, we use the augmented batch to update our model for a specified number of epochs before making the final prediction. After processing one test sample, the model is reset to its state from training.
To performs self-adaptation, add following parameters to the previously mentioned TTA arguments. We used our development set `WildDash` for hyperparameter tuning:
- ```--base-lr```: Learning rate for training on augmented batch (default: ```0.05```).
- ```--weight-decay```: Weight decay (default: ```0.0```).
- ```--momentum```: Momentum (default: ```0.0```).
- ```--num-epochs```: Numbers of epochs for each augmented batch (default:```10```).
- ```--threshold```: Ignore low-confidence predictions (default:```0.7```).
- ```--resnet-layers```: Layers 1, 2, 3 and/or 4 (corresponding to conv2_x - conv5_x) which will be frozen for self-adaptation (default: ```1 2```)
- ```--hrnet-layers```: Layers 1, 2 and/or 3 which will be frozen for self-adaptation (default: ```1 2```)

```
python eval.py --dataset-root DATASET-ROOT --source [gta|synthia] --checkpoints-root CHECKPOINT-ROOT --checkpoint CHECKPOINT --backbone-name [resnet50|resnet101] --num-classes [19|16] --batch-size 1 --scales 0.25 0.5 0.75 --threshold 0.7 --base-lr 0.05 --num-epochs 10 --flips --grayscale --num-workers 8 --weight-decay 0.0 --momentum 0.0
```

## Results

Our method achieves the following IoU for:

**Source Domain: GTA**

|Backbone      | Arch.         | Cityscapes | BDD    | IDD    | Mapillary | Checkpoint         |
|--------------|---------------|------------|--------|--------|-----------|--------------------|
|*ResNet50*    | DeepLabv1     | 45.13%     | 39.61% | 40.32% | 47.49%    | [resnet50_gta_alpha_0.1.pth](https://drive.google.com/uc?export=download&confirm=aOXz&id=1U1NFKvMQq_CxR26QtaI5vV5Y9hutixMf)    |
|*ResNet101*   | DeepLabv1     | 46.99%     | 40.21% | 40.56% | 47.49%    | [resnet101_gta_alpha_0.2.pth ](https://drive.google.com/uc?export=download&confirm=aOXz&id=1jbGfXJNLwjgPswZYKmmRExpUyEgDH9_p)  |

**Source Domain: SYNTHIA**

|Backbone      | Arch.         | Cityscapes | BDD    | IDD    | Mapillary | Checkpoint         |
|--------------|---------------|------------|--------|--------|-----------|--------------------|
|*ResNet50*    | DeepLabv1     | 41.60%     | 33.35% | 31.22% | 41.21%    | [resnet50_synthia_alpha_0.1.pth](https://drive.google.com/uc?export=download&confirm=aOXz&id=1FJT5trJsr-6e2fnH-VLYcjQQ-ZjRzJAC)    |
|*ResNet101*   | DeepLabv1     | 42.32%     | 33.27% | 31.40% | 41.20%    | [resnet101_synthia_alpha_0.1.pth](https://drive.google.com/uc?export=download&confirm=aOXz&id=11LI_IsCrJCsyZbRXKfpR3_d8NG0lq6tr)   |
