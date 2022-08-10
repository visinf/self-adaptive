import pathlib, os
from torch.utils.data import DataLoader
from torch.nn import SyncBatchNorm
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile

from utils.parser import train_parser
import models.backbone
from loss.semantic_seg import CrossEntropyLoss
import datasets
from optimizer.schedulers import *
from utils.metrics import *
from utils.distributed import init_process, clean_up
from utils import transforms
from utils.self_adapt_norm import reinit_alpha

import torch.distributed
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

# We set a maximum image size which can be fit on the GPU, in case the image is larger, we first downsample it
# to then upsample the prediction back to the original resolution. This is especially required for high resolution
# Mapillary images
img_max_size = (1024, 2048)


def main(opts):
    # Force disable distributed
    opts.distributed = False if not torch.cuda.is_available() else opts.distributed

    # Distributed training with multiple gpus
    if opts.distributed:
        opts.batch_size = opts.batch_size // opts.gpus
        mp.spawn(train,
                 nprocs=opts.gpus,
                 args=(opts,))

    # DataParallel with GPUs or CPU
    else:
        train(gpu=0, opts=opts)


def train(gpu: int,
          opts):
    # Create checkpoints directory
    pathlib.Path(opts.checkpoints_root).mkdir(parents=True, exist_ok=True)

    # Setup dataset
    # Get target domain from dataset path
    target_train = os.path.basename(opts.dataset_root)
    target_val = os.path.basename(opts.val_dataset_root)
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(opts.crop_size),
                                           transforms.RandomHFlip(),
                                           transforms.RandGaussianBlur(),
                                           transforms.ColorJitter(),
                                           transforms.MaskGrayscale(),
                                           transforms.ToTensor(),
                                           transforms.IdsToTrainIds(source=target_train, target=target_train),
                                           transforms.Normalize()])
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.IdsToTrainIds(source=target_train, target=target_val),
                                         transforms.ImgResize(img_max_size),
                                         transforms.Normalize()])

    train_dataset = datasets.__dict__[target_train](root=opts.dataset_root,
                                                   split="train",
                                                   transforms=train_transforms)
    val_dataset = datasets.__dict__[target_val](root=opts.val_dataset_root,
                                                split="val",
                                                transforms=val_transforms)
    # Setup model
    model = models.__dict__[opts.arch_type](backbone_name=opts.backbone_name,
                                            num_classes=opts.num_classes,
                                            alpha=opts.alpha,
                                            dropout=opts.dropout,
                                            update_source_bn=True)

    if opts.distributed:
        # Initialize process group
        rank = init_process(opts, gpu)

        # Convert batch normalization to SyncBatchNorm and setup CUDA
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        torch.cuda.set_device(gpu)
        model.cuda(gpu)

        # Wrap model in DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[gpu], find_unused_parameters=True)

        # Setup data sampler and loader
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=opts.world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(dataset=val_dataset, num_replicas=opts.world_size, rank=rank, shuffle=False)
    else:
        # Run on GPU if available else on CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.nn.DataParallel(model).to(device)
        train_sampler = None
        val_sampler = None

    # Set main process and device
    main_process = not opts.distributed or (opts.distributed and rank == 0)
    device = gpu if opts.distributed else device

    # Add tensorboard writer and setup metric
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if main_process:
        print(f"Current training run {time_stamp} has started!")
        iou_meter = runningScore(opts.num_classes)
        alphas = np.round(np.linspace(0, 1, opts.num_alphas), 5) if opts.num_alphas > 1 else [opts.alpha]

    # Setup dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=opts.batch_size,
                              num_workers=opts.num_workers,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              pin_memory=True if torch.cuda.is_available() else False)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=opts.num_workers,
                            sampler=val_sampler,
                            shuffle=False,
                            pin_memory=True if torch.cuda.is_available() else False)

    # Setup loss
    criterion = CrossEntropyLoss().to(device)

    # Setup lr scheduler, optimizer and loss
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opts.base_lr,
                                momentum=opts.momentum,
                                weight_decay=opts.weight_decay)

    scheduler = get_scheduler(scheduler_type=opts.lr_scheduler,
                              optimizer=optimizer,
                              max_iter=len(train_loader) * opts.num_epochs + 1)

    # Training
    mean_iou_best_alphas = [0] * opts.num_alphas
    model.train()
    for epoch in tqdm(range(opts.num_epochs)):
        if opts.distributed:
            train_sampler.set_epoch(epoch)

        for train_idx, (img_train, gt_train) in enumerate(train_loader):

            # Put img and gt on GPU if available
            img_train, gt_train = img_train.to(device), gt_train.to(device)

            # Forward pass, backward pass and optimization
            out_train = model(img=img_train)
            loss_train = criterion(out_train['pred'], gt_train)

            # Zero the parameter gradients
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        if epoch >= opts.validation_start and epoch % opts.validation_step == 0:
            if main_process:
                # Set model to eval
                model.eval()
                with torch.no_grad():
                    score_alphas, class_iou_epoch_alphas = [], []
                    for alpha_idx, alpha in enumerate(alphas):
                        reinit_alpha(model, alpha, device)
                        for val_idx, (img_val, gt_val) in enumerate(val_loader):
                            # Put img and gt on GPU if available
                            img_val, gt_val = img_val.to(device), gt_val.to(device)
                            # Forward pass and loss calculation
                            out_val = model(img=img_val)['pred']
                            # Upsample prediction to gt resolution
                            out_val = torch.nn.functional.interpolate(out_val,
                                                                      size=gt_val.shape[-2:],
                                                                      mode='bilinear')
                            # Update iou meter
                            iou_meter.update(gt_val.cpu().numpy(), torch.argmax(out_val, dim=1).cpu().numpy())

                        score, class_iou_epoch, _, _ = iou_meter.get_scores()
                        mean_iou_epoch = score['Mean IoU :']
                        score_alphas.append(mean_iou_epoch)
                        iou_meter.reset()

                        # Save model if mean iou higher than before
                        if mean_iou_epoch > mean_iou_best_alphas[alpha_idx]:
                            checkpoints_path = os.path.join(opts.checkpoints_root,
                                                            time_stamp + f'_alpha_{alpha}.pth')
                            if os.path.isfile(checkpoints_path):
                                os.remove(checkpoints_path)
                            torch.save(model.state_dict(), checkpoints_path)
                            mean_iou_best_alphas[alpha_idx] = mean_iou_epoch

            # Switch model to train
            model.train()

        # Final result
        if main_process and epoch == opts.num_epochs - 1:
            print(f"alphas: {[i for i in alphas]}:")
            print(f"IoUs: {mean_iou_best_alphas}")
            checkpoints_path = os.path.join(opts.checkpoints_root, time_stamp + '.pth')
            if os.path.isfile(checkpoints_path):
                os.remove(checkpoints_path)
            alpha_ind_max = torch.argmax(torch.tensor(mean_iou_best_alphas)).item()
            alpha = alphas[alpha_ind_max]
            checkpoints_alpha_path = os.path.join(opts.checkpoints_root,
                                                  time_stamp + f'_alpha_{alpha}.pth')
            copyfile(checkpoints_alpha_path, checkpoints_path)
            print(f"Saved checkpoint based on alpha = {alpha}")
            print(f"Current training run {time_stamp} is finished!")

    if opts.distributed:
        clean_up()

if __name__ == '__main__':
    args = train_parser()
    print(args)
    main(args)