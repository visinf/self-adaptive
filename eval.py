import glob
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.parser import val_parser
from loss.semantic_seg import CrossEntropyLoss
import models.backbone
import models
from utils.modeling import freeze_layers
from utils.self_adapt_norm import reinit_alpha
from utils.metrics import *
from utils.calibration import *
from datasets.labels import *
from datasets.self_adapt_augment import TrainTestAugDataset
torch.backends.cudnn.benchmark = True

# We set a maximum image size which can be fit on the GPU, in case the image is larger, we first downsample it
# to then upsample the prediction back to the original resolution. This is especially required for high resolution
# Mapillary images
img_max_size = [1024, 2048]


def main(opts):
    # Setup metric
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    iou_meter = runningScore(opts.num_classes)
    print(f"Current inference run {time_stamp} has started!")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup dataset and transforms
    test_dataset = TrainTestAugDataset(device=device,
                                       root=opts.dataset_root,
                                       only_inf=opts.only_inf,
                                       source=opts.source,
                                       crop_size=img_max_size,
                                       split=opts.dataset_split,
                                       threshold=opts.threshold,
                                       tta=opts.tta,
                                       flips=opts.flips,
                                       scales=opts.scales,
                                       grayscale=opts.grayscale)
    test_loader = DataLoader(test_dataset,
                             batch_size=opts.batch_size,
                             shuffle=False,
                             num_workers=opts.num_workers)

    # Load and setup model
    model = models.__dict__[opts.arch_type](backbone_name=opts.backbone_name,
                                            num_classes=opts.num_classes,
                                            update_source_bn=False,
                                            dropout=opts.dropout)
    model = torch.nn.DataParallel(model)

    # Pick newest checkpoints
    if os.path.exists(opts.checkpoints_root):
        checkpoint = max(glob.glob(os.path.join(opts.checkpoints_root, opts.checkpoint)), key=os.path.getctime)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        # Reinitialize alpha if a custom alpha other than the one in the checkpoints is given
        if opts.alpha is not None:
            reinit_alpha(model, alpha=opts.alpha, device=device)
    else:
        raise ValueError(f"Checkpoints directory {opts.checkpoints_root} does not exist")

    model = model.to(device)

    # Set up Self-adaptive learning optimizer and loss
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opts.base_lr,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay
    )
    criterion = CrossEntropyLoss().to(device)

    if opts.calibration:
        # Calibration meter
        cal_meter = CalibrationMeter(
            device,
            n_bins=10,
            num_classes=opts.num_classes,
            num_images=len(test_loader)
        )
    model.eval()

    # Create GradScaler for mixed precision
    if opts.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    for test_idx, (img_test, gt_test, crop_test, crop_transforms) in enumerate(tqdm(test_loader)):
        # Put img on GPU if available
        img_test = img_test.to(device)
        if opts.only_inf:
            # Forward pass with original image
            with torch.no_grad():
                if opts.mixed_precision:
                    with torch.cuda.amp.autocast():
                        out_test = model(img=img_test)['pred']
                else:
                    out_test = model(img=img_test)['pred']
        else:
            # Reload checkpoints
            model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
            # Reinitialize alpha if a custom alpha other than the one in the checkpoints is given
            if opts.alpha is not None:
                reinit_alpha(model, alpha=opts.alpha, device=device)

            model = model.to(device)

            # Compute augmented predictions
            crop_test_fused = []
            for crop_test_sub in crop_test:
                with torch.no_grad():
                    if opts.mixed_precision:
                        with torch.cuda.amp.autocast():
                            out_test = model(img=crop_test_sub)['pred']
                    else:
                        out_test = model(img=crop_test_sub)['pred']
                crop_test_fused.append(torch.nn.functional.softmax(out_test, dim=1))

            # Create pseudo gt from augmentations based on their softmax probabilities
            pseudo_gt = test_dataset.create_pseudo_gt(
                crop_test_fused, crop_transforms, [1, opts.num_classes, *img_test.shape[-2:]]
            )
            pseudo_gt = pseudo_gt.to(device)

            if opts.tta:
                # Use pseudo gt for evaluation
                out_test = pseudo_gt
            else:
                model.train()

                # Freeze layers if given
                freeze_layers(opts, model)

                # Self-adaptive learning loop
                model = model.to(device)
                for epoch in range(opts.num_epochs):
                    if opts.mixed_precision:
                        with torch.cuda.amp.autocast():
                            out_test = model(img=img_test)['pred']
                    else:
                        out_test = model(img=img_test)['pred']
                    if opts.mixed_precision:
                        with torch.cuda.amp.autocast():
                            loss_train = criterion(out_test, pseudo_gt)
                    else:
                        loss_train = criterion(out_test, pseudo_gt)
                    optimizer.zero_grad()
                    if opts.mixed_precision:
                        scaler.scale(loss_train).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss_train.backward()
                        optimizer.step()

                # Do actual forward pass with updated model
                model.eval()
                with torch.no_grad():
                    if opts.mixed_precision:
                        with torch.cuda.amp.autocast():
                            out_test = model(img=img_test)['pred']
                    else:
                        out_test = model(img=img_test)['pred']

        # Upsample prediction to gt resolution
        out_test = torch.nn.functional.interpolate(out_test, size=gt_test.shape[-2:], mode='bilinear')

        # Update calibration meter
        if opts.calibration:
            cal_meter.calculate_bins(out_test, gt_test.to(device))

        # Add prediction
        iou_meter.update(gt_test.cpu().numpy(), torch.argmax(out_test, dim=1).cpu().numpy())

    # Save output
    score, _, _, _ = iou_meter.get_scores()
    mean_iou = score['Mean IoU :']

    # Compute ECE
    if opts.calibration:
        cal_meter.calculate_mean_over_dataset()
        print(f"ECE: {cal_meter.overall_ece}")

    print(f"Mean IoU: {mean_iou}")
    print(f"Current inference run {time_stamp} is finished!")

if __name__ == '__main__':
    args = val_parser()
    print(args)
    main(args)