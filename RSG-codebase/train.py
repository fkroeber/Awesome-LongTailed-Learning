import argparse
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from losses import LDAMLoss
from collections import OrderedDict
import models
from utils import (
    ImbalancedDatasetSampler,
    accuracy,
    macro_f1_score,
    prepare_folders,
    save_checkpoint,
)
from data_imagenet import ImageNet_LT
from data_s2 import S2_Dataset

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--dataset", default="imagenet", choices=["imagenet", "s2"])
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnext50_32x4d",
    choices=model_names,
    help="model architecture: "
    + " | ".join(model_names)
    + " (default: resnext50_32x4d)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--fold",
    default=0,
    type=int,
    metavar="N",
    help="fold for s2 dataset (default: 0)",
    dest="fold",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 1)",
    dest="print_freq",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for initializing training. ",
    dest="seed",
)
parser.add_argument("--root_path", type=str, default="./data")
parser.add_argument("--mark", default=None, type=str, help="log dir")
parser.add_argument("--image_dir", type=str, default="../data/ImageNet")
parser.add_argument("--head_tail_ratio", type=float, default=0.2)

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.epoch_thresh = int(
        args.epochs * 0.75
    )  # point from which on samples are generated
    args.store_name = "_".join(["ImageNet_LT", args.arch])
    args.root_log = f"{args.root_path}/{args.dataset}/{args.mark}"
    args.root_model = f"{args.root_path}/{args.dataset}/{args.mark}"
    os.makedirs(args.root_log, exist_ok=True)
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    main_worker(args)


def main_worker(args):
    global best_acc1

    print("=> creating model '{}'".format(args.arch))

    # Data loading code

    if args.dataset == "imagenet":
        args.num_classes = 1000
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = ImageNet_LT(args.image_dir, transform_train, "train")
        val_dataset = ImageNet_LT(args.image_dir, transform_val, "val")

    elif args.dataset == "s2":
        args.num_classes = 2
        mean = [
            0.0239,
            0.0486,
            0.0349,
            0.0867,
            0.2398,
            0.2892,
            0.3028,
            0.3206,
            0.1605,
            0.0816,
            0.0189,
            0.0426,
            0.0297,
            0.0795,
            0.2427,
            0.2977,
            0.3098,
            0.3300,
            0.1662,
            0.0819,
            0.0209,
            0.0429,
            0.0370,
            0.0798,
            0.2162,
            0.2649,
            0.2778,
            0.2983,
            0.1791,
            0.0920,
            0.0222,
            0.0429,
            0.0432,
            0.0812,
            0.1958,
            0.2388,
            0.2516,
            0.2736,
            0.1884,
            0.1003,
            0.0193,
            0.0364,
            0.0295,
            0.0662,
            0.1838,
            0.2260,
            0.2372,
            0.2585,
            0.1634,
            0.0827,
        ]
        std = [
            0.0157,
            0.0194,
            0.0264,
            0.0266,
            0.0628,
            0.0782,
            0.0892,
            0.0852,
            0.0485,
            0.0349,
            0.0159,
            0.0200,
            0.0262,
            0.0279,
            0.0579,
            0.0741,
            0.0859,
            0.0809,
            0.0500,
            0.0344,
            0.0172,
            0.0224,
            0.0320,
            0.0346,
            0.0484,
            0.0613,
            0.0726,
            0.0683,
            0.0680,
            0.0459,
            0.0198,
            0.0258,
            0.0414,
            0.0412,
            0.0437,
            0.0557,
            0.0670,
            0.0624,
            0.0831,
            0.0576,
            0.0171,
            0.0239,
            0.0309,
            0.0360,
            0.0620,
            0.0729,
            0.0832,
            0.0813,
            0.0818,
            0.0521,
        ]
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                MultiChannelColorJitter(brightness=0.4, contrast=0.4),
                transforms.Normalize(mean, std),
            ]
        )
        transform_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(mean, std),
            ]
        )
        train_dataset = S2_Dataset(
            meta_file="/nfsdata/data/grid/grid_vulkaneifel_sentinel_2560_0_folds.geojson",
            phase="train",
            fold=args.fold,
            data_usage=1,
            transform=transform_train,
        )
        val_dataset = S2_Dataset(
            meta_file="/nfsdata/data/grid/grid_vulkaneifel_sentinel_2560_0_folds.geojson",
            phase="val",
            fold=args.fold,
            data_usage=1,
            transform=transform_val,
        )

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    try:
        cls_num_list = train_dataset.cls_num_list  # for S2
    except:
        cls_num_list = train_dataset.get_cls_num_list()  # for ImageNet_LT

    print("cls num list:")
    print(cls_num_list)

    args.cls_num_list = cls_num_list.copy()

    head_lists = []
    Inf = 0
    # identify at least one head class
    for i in range(max(int(args.num_classes * args.head_tail_ratio), 1)):
        head_lists.append(cls_num_list.index(max(cls_num_list)))
        cls_num_list[cls_num_list.index(max(cls_num_list))] = Inf

    model = models.__dict__[args.arch](
        num_classes=args.num_classes,
        head_lists=head_lists,
        phase_train=True,
        epoch_thresh=args.epoch_thresh,
    )
    model = torch.nn.DataParallel(model).cuda()
    print(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    cudnn.benchmark = True
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # init log for training
    with open(os.path.join(args.root_log, f"{args.mark}_args.txt"), "w") as f:
        f.write(str(args))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if epoch == args.epoch_thresh:
            train_sampler = ImbalancedDatasetSampler(
                train_dataset, label_count=args.cls_num_list
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
            )

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Cls num list: {args.cls_num_list}")
        effective_num = 1.0 - np.power(0, args.cls_num_list)
        print(f"Effective num: {effective_num}")
        per_cls_weights = (1.0) / np.array(effective_num)
        per_cls_weights = (
            per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num_list)
        )
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        print(f"Per class weights: {per_cls_weights}")

        criterion = LDAMLoss(
            cls_num_list=args.cls_num_list, max_m=0.3, s=30, weight=per_cls_weights
        ).cuda()

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
        )

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        output_best = "Best Prec@1: %.3f\n" % (best_acc1)
        print(output_best)

        # remove RSG module, since RSG is not used during testing.
        new_state_dict = OrderedDict()
        for k in model.state_dict().keys():
            name = k[7:]  # remove `module.`
            if "RSG" in k:
                continue
            new_state_dict[name] = model.state_dict()[k]

        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": new_state_dict,
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    f1_meter = AverageMeter("F1", ":6.3f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, f1_meter],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    all_preds = []
    all_targets = []
    end = time.time()
    for i, (input, target) in enumerate(iter(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        # compute output
        output, cesc_loss, total_mv_loss, combine_target = model(input, epoch, target)
        ldam_loss = criterion(output, combine_target)
        loss = ldam_loss + 0.1 * cesc_loss.mean() + 0.01 * total_mv_loss.mean()
        print(
            f"LDAM Loss: {ldam_loss.mean().item():.4f}, CESC Loss: {0.1*cesc_loss.mean().item():.4f}, Total MV Loss: {0.01*total_mv_loss.mean().item():.4f}"
        )

        acc1 = accuracy(output, combine_target, topk=(1,))
        losses.update(loss.item(), output.size(0))
        top1.update(acc1[0], output.size(0))

        # F1 tracking
        preds = torch.argmax(output, dim=1)
        all_preds.append(preds.detach())
        all_targets.append(combine_target.detach())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # compute F1 so far
            y_true = torch.cat(all_targets)
            y_pred = torch.cat(all_preds)
            f1 = macro_f1_score(y_true, y_pred, args.num_classes)
            f1_meter.update(f1, y_true.size(0))
            # print F1 score
            progress.display(i, args)


def validate(val_loader, model, criterion, epoch, args, flag="val"):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    f1_meter = AverageMeter("F1", ":6.3f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, f1_meter], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    total_logits = torch.empty((0, args.num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # if args.gpu is not None:
            input = input.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            # compute output
            output = model(input, phase_train=False)
            loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))

            # F1 tracking
            preds = torch.argmax(output, dim=1)
            all_preds.append(preds.detach())
            all_targets.append(target.detach())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # Compute F1 so far
        y_true = torch.cat(all_targets)
        y_pred = torch.cat(all_preds)
        f1 = macro_f1_score(y_true, y_pred, args.num_classes)
        f1_meter.update(f1, y_true.size(0))
        progress.display(i, args)

        progress.display(len(val_loader), args)
        open(args.root_log + "/" + args.mark + ".log", "a+").write("\n")

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch > int(0.9 * args.epochs) - 1:
        lr = args.lr * 0.001
    elif epoch > int(0.75 * args.epochs) - 1:
        lr = args.lr * 0.01
    elif epoch > int(0.6 * args.epochs) - 1:
        lr = args.lr * 0.1
    else:
        if epoch <= 5:
            lr = args.lr * epoch / 5
        else:
            lr = args.lr

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        d = self.__dict__.copy()
        for key in d.keys():
            if isinstance(d[key], torch.Tensor):
                d[key] = d[key].item()
        return fmtstr.format(**d)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        open(args.root_log + "/" + args.mark + ".log", "a+").write(
            "\t".join(entries) + "\n"
        )

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class MultiChannelColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        """
        Args:
            img (Tensor): image tensor of shape (C, H, W), not PIL!
        """
        # Apply brightness jitter
        if self.brightness > 0:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            img = img * brightness_factor

        # Apply contrast jitter
        if self.contrast > 0:
            mean = img.mean(dim=(1, 2), keepdim=True)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            img = (img - mean) * contrast_factor + mean

        return img


if __name__ == "__main__":
    main()
