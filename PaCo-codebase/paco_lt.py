import argparse
import builtins
import math
import os

os.environ["OPENBLAS_NUM_THREADS"] = "2"
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from models import resnet_imagenet, resnet_s2
from randaugment import rand_augment_transform, GaussianBlur
import moco.loader
import moco.builder
from dataset.imagenet import ImageNetLT
from dataset.imagenet_moco import ImageNetLT_moco
from dataset.s2_dataset import S2_Dataset, S2_Dataset_moco
from losses import PaCoLoss
from utils import shot_acc
from sklearn.metrics import f1_score

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names += ["resnext101_32x4d"]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="imagenet", choices=["inat", "imagenet", "s2"])
parser.add_argument("--data", metavar="DIR", default="../data/ImageNet")
parser.add_argument("--root_path", type=str, default="./data")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnext50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=400, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
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
    "--fold",
    default=0,
    type=int,
    metavar="FOLD",
    help="fold for cross-val (only S2)",
    dest="fold",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=1,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank", default=0, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://localhost:9996",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    default=True,
    type=bool,
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--moco-k",
    default=8192,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.2, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", default=True, type=bool, help="use mlp head")
parser.add_argument(
    "--aug-plus", default=True, type=bool, help="use moco v2 data augmentation"
)
parser.add_argument("--cos", default=True, type=bool, help="use cosine lr schedule")
parser.add_argument(
    "--normalize", default=False, type=bool, help="use cosine lr schedule"
)

# options for paco
parser.add_argument("--mark", default=None, type=str, help="log dir")
parser.add_argument("--reload", default=None, type=str, help="load supervised model")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument(
    "--alpha", default=1.0, type=float, help="contrast weight among samples"
)
parser.add_argument(
    "--beta",
    default=1.0,
    type=float,
    help="contrast weight between centers and samples",
)
parser.add_argument("--gamma", default=1.0, type=float, help="paco loss")
parser.add_argument("--aug", default=None, type=str, help="aug strategy")
parser.add_argument("--randaug_m", default=10, type=int, help="randaug-m")
parser.add_argument("--randaug_n", default=2, type=int, help="randaug-n")
parser.add_argument(
    "--num_classes", default=1000, type=int, help="num classes in dataset"
)
parser.add_argument(
    "--feat_dim", default=2048, type=int, help="last feature dim of backbone"
)

# fp16
parser.add_argument("--fp16", action="store_true", help=" fp16 training")


best_acc = 0


def main():
    args = parser.parse_args()
    args.root_model = f"{args.root_path}/{args.dataset}/{args.mark}"
    os.makedirs(args.root_model, exist_ok=True)

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

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == "s2":
        _model = getattr(resnet_s2, args.arch)
    else:
        _model = getattr(resnet_imagenet, args.arch)

    model = moco.builder.MoCo(
        _model,
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        args.mlp,
        args.feat_dim,
        args.normalize,
        num_classes=args.num_classes,
    )
    print(model)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = PaCoLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.moco_t,
        K=args.moco_k,
        num_classes=args.num_classes,
    ).cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    txt_train = "./imagenet/data/ImageNet_LT/ImageNet_LT_train.txt"
    txt_test = "./imagenet/data/ImageNet_LT/ImageNet_LT_test.txt"

    if args.dataset == "inat":
        normalize = transforms.Normalize(
            mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]
        )
    elif args.dataset == "s2":
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
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [MultiChannelColorJitter(brightness=0.4, contrast=0.4)],
                p=0.8,  # not strengthened
            ),
            MultiChannelRandomDesaturation(p=0.2),
            # transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            MultiChannelRandomDesaturation(p=0.2),
            MultiChannelColorJitter(brightness=0.4, contrast=0.4),
            transforms.RandomHorizontalFlip(),
            normalize,
        ]

    augmentation_regular = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        MultiChannelColorJitter(brightness=0.4, contrast=0.4),
        normalize,
    ]

    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomApply(
            [MultiChannelColorJitter(brightness=0.4, contrast=0.4)],
            p=1.0,  # not strengthened
        ),
        MultiChannelRandomDesaturation(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]

    augmentation_sim02 = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [MultiChannelColorJitter(brightness=0.4, contrast=0.4)],
            p=1.0,  # not strengthened
        ),
        MultiChannelRandomDesaturation(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(
        translate_const=int(224 * 0.45),
        img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
    )
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [MultiChannelColorJitter(brightness=0.4, contrast=0.4)], p=1.0
        ),
        MultiChannelRandomDesaturation(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        rand_augment_transform(
            "rand-n{}-m{}-mstd0.5".format(args.randaug_n, args.randaug_m), ra_params
        ),
        normalize,
    ]

    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [MultiChannelColorJitter(brightness=0.4, contrast=0.4)], p=1.0
        ),
        rand_augment_transform(
            "rand-n{}-m{}-mstd0.5".format(args.randaug_n, args.randaug_m), ra_params
        ),
        normalize,
    ]

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            normalize,
        ]
    )

    if args.dataset == "imagenet":
        val_dataset = ImageNetLT(root=args.data, txt=txt_test, transform=val_transform)
    elif args.dataset == "s2":
        val_dataset = S2_Dataset(
            meta_file="/nfsdata/data/grid/grid_vulkaneifel_sentinel_2560_0_folds.geojson",
            phase="val",
            fold=args.fold,
            data_usage=1,
            transform=val_transform,
        )
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))

    if args.aug == "regular_regular":
        transform_train = [
            transforms.Compose(augmentation_regular),
            transforms.Compose(augmentation),
        ]
    elif args.aug == "mocov2_mocov2":
        transform_train = [
            transforms.Compose(augmentation),
            transforms.Compose(augmentation),
        ]
    elif args.aug == "sim_sim":
        transform_train = [
            transforms.Compose(augmentation_sim),
            transforms.Compose(augmentation_sim),
        ]
    elif args.aug == "randcls_sim":
        transform_train = [
            transforms.Compose(augmentation_randncls),
            transforms.Compose(augmentation_sim),
        ]
    elif args.aug == "randclsstack_sim":
        transform_train = [
            transforms.Compose(augmentation_randnclsstack),
            transforms.Compose(augmentation_sim),
        ]
    elif args.aug == "randclsstack_sim02":
        transform_train = [
            transforms.Compose(augmentation_randnclsstack),
            transforms.Compose(augmentation_sim02),
        ]

    if args.dataset == "imagenet":
        train_dataset = ImageNetLT_moco(
            root=args.data, txt=txt_train, transform=transform_train
        )
    elif args.dataset == "s2":
        train_dataset = S2_Dataset_moco(
            meta_file="/nfsdata/data/grid/grid_vulkaneifel_sentinel_2560_0_folds.geojson",
            phase="train",
            fold=args.fold,
            data_usage=1,
            transform=transform_train,
        )
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    print(f"===> Training data length {len(train_dataset)}")

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    train_sampler = None

    criterion.cal_weight_for_classes(train_dataset.cls_num_list)

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        print(" start evaualteion **** ")
        validate(val_loader, train_loader, model, criterion_ce, args)
        return

    # mixed precision
    scaler = GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scaler, args)
        acc = validate(val_loader, train_loader, model, criterion_ce, args)
        if acc > best_acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "acc": acc,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=is_best,
                filename=f"{args.root_model}/moco_ckpt.pth.tar",
            )


def train(train_loader, model, criterion, optimizer, epoch, scaler, args):
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

    end = time.time()
    all_preds = []
    all_targets = []
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if not args.fp16:
            features, labels, logits = model(
                im_q=images[0], im_k=images[1], labels=target
            )
            loss = criterion(features, labels, logits)
        else:
            with autocast():
                features, labels, logits = model(
                    im_q=images[0], im_k=images[1], labels=target
                )
                loss = criterion(features, labels, logits)

        acc1 = accuracy(logits, target, topk=(1,))
        losses.update(loss.item(), logits.size(0))
        top1.update(acc1[0], logits.size(0))

        # F1 tracking
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach())
        all_targets.append(target.detach())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if not args.fp16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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


def validate(val_loader, train_loader, model, criterion, args):
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
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            total_logits = torch.cat((total_logits, output))
            total_labels = torch.cat((total_labels, target))

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

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
        open(args.root_model + "/" + args.mark + ".log", "a+").write("\n")

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))


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
        open(args.root_model + "/" + args.mark + ".log", "a+").write(
            "\t".join(entries) + "\n"
        )

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs + 1)
                / (args.epochs - args.warmup_epochs + 1)
            )
        )
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def macro_f1_score(y_true, y_pred, num_classes):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))


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


class MultiChannelRandomDesaturation:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor of shape (C, H, W)
        """
        if random.random() < self.p:
            # Average across channels and expand to original shape
            gray = img.mean(dim=0, keepdim=True)  # shape: (1, H, W)
            img = gray.expand_as(img)  # broadcast to (C, H, W)
        return img


if __name__ == "__main__":
    main()
