from .voc import VOCSegmentation, VOCSegmentationIncremental, VOCasCOCOSegmentationIncremental, VOCFSSSegmentationIncremental
from .coco import COCO, COCOIncremental, COCOFSSegmentationIncremental
from .cub import CUB200SegmentationIncremental
from .transform import *
import tasks
import os
from torch import distributed
import random
from .dataset import ImageNetDataset


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transform.Compose([
        transform.Resize(size=opts.crop_size_val),
        transform.CenterCrop(size=opts.crop_size_val),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    test_transform = val_transform

    ext_dset_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 1.0)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    step_dict = tasks.get_task_dict(opts.dataset, opts.task, opts.step)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)

    pseudo = f"{opts.pseudo}_{opts.task}_{opts.step}" if opts.pseudo is not None else None
    path_base = os.path.join(opts.data_root, path_base)

    labels_cum = labels_old + labels
    masking_value = 0

    if opts.dataset == 'voc':
        t_dataset = dataset = VOCSegmentationIncremental
    elif opts.dataset == 'coco':
        t_dataset = dataset = COCOIncremental
    elif opts.dataset == 'coco-voc':
        if opts.step == 0:
            t_dataset = dataset = COCOIncremental
        else:
            dataset = VOCasCOCOSegmentationIncremental
            t_dataset = COCOIncremental
    else:
        raise NotImplementedError

    if opts.overlap and opts.dataset == 'voc':
        path_base += "-ov"

    path_base_train = path_base

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)
    
    train_idxs_path = list()
    if opts.replay:
        for i in range(0, opts.step + 1):
            train_idxs_path.append(path_base_train + f"/train-{i}.npy")
    else:
        train_idxs_path.append(path_base_train + f"/train-{opts.step}.npy")
    
    train_dst = dataset(root=opts.data_root, step_dict=step_dict, train=True, transform=train_transform,
                        idxs_path=train_idxs_path, masking_value=masking_value,
                        masking=not opts.no_mask, overlap=opts.overlap, step=opts.step, weakly=opts.weakly,
                        saliency=False
                        #saliency=opts.saliency
                        , pseudo=pseudo, replay=opts.replay, replay_size=opts.replay_size)
    
    if opts.external_dataset:
        external_dst = ImageNetDataset(root=opts.external_rootdir, imagenet_dict=train_dst.imagenet_wnid_dict(),
                                       step_dict=step_dict, mapping_dict=train_dst.inverted_order, 
                                       step=opts.step, external_size=opts.external_size, external_mode=opts.external_mode,
                                       labels_cum=labels_cum, transform=ext_dset_transform, split_root=path_base_train)

    # Val is masked with 0 when label is not known or is old (masking=True, masking_value=0)
    val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform,
                      idxs_path=path_base + f"/val-{opts.step}.npy", masking_value=masking_value,
                      masking=False, overlap=opts.overlap, step=opts.step, weakly=opts.weakly)

    # Test is masked with 255 for labels not known and the class for old (masking=False, masking_value=255)
    image_set = 'train' if opts.val_on_trainset else 'val'
    test_dst = t_dataset(root=opts.data_root, step_dict=step_dict, train=opts.val_on_trainset, transform=test_transform,
                         masking=False, masking_value=255, weakly=opts.weakly,
                         idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy", step=opts.step)

    if opts.external_dataset:
        return train_dst, val_dst, test_dst, labels_cum, len(labels_cum), external_dst
    else:
        return train_dst, val_dst, test_dst, labels_cum, len(labels_cum)

def get_fss_dataset(opts, task):
    # Dataset and Augmentation for few-shot experiments

    train_transform = transform.Compose([
        transform.RandomScale((0.5, 2)),
        transform.RandomCrop(opts.crop_size, pad_if_needed=True),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transform.Compose([
        transform.PadCenterCrop(size=opts.crop_size_test),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.dataset == 'voc':
        dataset = VOCFSSSegmentationIncremental
    elif opts.dataset == 'coco':
        dataset = COCOFSSegmentationIncremental
    else:
        raise NotImplementedError(f'The dataset {opts.dataset} is not yet supported.')
    
    train_dst = dataset(root=opts.data_root, task=task, train=True, transform=train_transform, masking=True)
    val_dst = dataset(root=opts.data_root, task=task, train=False, transform=val_transform)
    test_dst = dataset(root=opts.data_root, task=task, train=False, transform=test_transform)

    return train_dst, val_dst, test_dst

def get_fgr_dataset(opts):
    # dataset for fine grained experiments

    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transform.Compose([
        transform.Resize(size=opts.crop_size_val),
        transform.CenterCrop(size=opts.crop_size_val),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    test_transform = val_transform
    
    step_dict = tasks.get_task_dict(opts.dataset, opts.task, opts.step)
    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    path_base = os.path.join(opts.data_root, path_base)

    labels_cum = labels_old + labels
    masking_value = 0

    if opts.dataset == 'cub':
        t_dataset = dataset = CUB200SegmentationIncremental
    else:
        raise NotImplementedError(f'The fine grained dataset {opts.dataset} is not yet supported.')

    train_dst = dataset(root=opts.data_root, step_dict=step_dict, train=True, transform=train_transform,
                        masking=True, masking_value=masking_value, step=opts.step)

    val_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=val_transform,
                      masking=False, masking_value=masking_value, step=opts.step)
    
    test_dst = dataset(root=opts.data_root, step_dict=step_dict, train=False, transform=test_transform,
                       masking=False, masking_value=255, step=opts.step)
    
    return train_dst, val_dst, test_dst, labels_cum, len(labels_cum)