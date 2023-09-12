from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import torch
from torch import distributed
from torchvision.utils import make_grid, save_image
import copy
import subprocess
import socket
import matplotlib.pyplot as plt
import os
import cv2

def label_to_one_hot(y, num_classes):
    y_tensor = y.view(y.shape[0], -1, y.shape[1], y.shape[2])
    zeros = torch.zeros(y.shape[0], num_classes, y.shape[1], y.shape[2], dtype=torch.float, device=y.device)
    return zeros.scatter(1, y_tensor, 1)


def clear_ckpt(checkpoint, remove=True):
    if remove:
        state = {}
        for k, v in checkpoint.items():
            state[k[7:]] = v
    else:
        state = checkpoint
    return state


def denorm(image, mean=(0.485, 0.456, 0.4069), std=(0.229, 0.224, 0.225)):
    image = image.clone()
    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [CxHxW]"
        assert image.size(0) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
    elif image.dim() == 4:
        # batch mode
        assert image.size(1) == 3, "Expected RGB image [3xHxW]"

        for t, m, s in zip((0, 1, 2), mean, std):
            image[:, t, :, :].mul_(s).add_(m)

    return image


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


def color_map(dataset):
    if dataset=='voc':
        return voc_cmap()
    elif dataset=='cityscapes':
        return cityscapes_cmap()
    elif dataset=='ade' or dataset=='coco' or dataset=='coco-voc':
        return ade_cmap()
    elif dataset == 'cub':
        return voc_cmap()


def cityscapes_cmap():
    return np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30), 
                         (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), 
                         (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0)], 
                         dtype=np.uint8)


def ade_cmap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255]
    ]

    for i in range(len(colors)):
        cmap[i] = colors[i]

    return cmap.astype(np.uint8)


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class Label2Color(object):
    def __init__(self, cmap):
        self.cmap = cmap

    def __call__(self, lbls):
        return self.cmap[lbls]

def visualize_external(dataset, ret_images, mapping_dict):
    # visualize images for step > 0
    cmap = color_map(dataset=dataset)
    lab2color = Label2Color(cmap)
    imgs = []
    preds = []
    preds_old = []
    loc_preds = []
    inverted_mapping_dict = {v: k for k, v in mapping_dict.items()}

    for i, (img, pred, loc_pred, pred_old) in enumerate(ret_images):
        img = torch.permute(denorm(img), (1, 2, 0)).unsqueeze(0)
        # transform model predictions to match the ground truth color maps
        pred_copy = copy.deepcopy(pred)
        pred_old_copy = copy.deepcopy(pred_old)
        loc_pred_copy = copy.deepcopy(loc_pred)

        for k, v in inverted_mapping_dict.items():
            pred_copy[pred == k] == v
            pred_old_copy[pred_old == k] = v
            loc_pred_copy[loc_pred == k] = v
        
        pred = torch.from_numpy(lab2color(pred_copy) / 255.).unsqueeze(0)
        pred_old = torch.from_numpy(lab2color(pred_old_copy) / 255.).unsqueeze(0)
        loc_pred = torch.from_numpy(lab2color(loc_pred_copy) / 255.).unsqueeze(0)

        imgs.append(img)
        preds.append(pred)
        preds_old.append(pred_old)
        loc_preds.append(loc_pred)
    
    imgs = torch.cat(imgs, dim=0)
    preds = torch.cat(preds, dim=0)
    preds_old = torch.cat(preds_old, dim=0)
    loc_preds = torch.cat(loc_preds, dim=0)

    concat_imgs = torch.cat((imgs, preds, loc_preds, preds_old), dim=2) # concat along width
    concat_imgs = torch.permute(concat_imgs, (0, 3, 1, 2)) # bs x c x h x w
    imgs_grid = make_grid(concat_imgs, nrow=4)
    return imgs_grid

def visualize_predictions(dataset, output, mapping_dict, base_path, idx):
    # to visualize
    cmap = color_map(dataset=dataset)
    lab2color = Label2Color(cmap)
    similarity_maps = []

    # invert the mapping dict to preserve the colors
    inverted_mapping_dict = {v: k for k, v in mapping_dict.items()}

    # get from the output dict
    img = output['image'].squeeze(0)
    lbl = output['label'].squeeze(0)
    loc_pred = output['loc_pred'].squeeze(0)
    main_pred = output['main_pred'].squeeze(0)
    old_pred = output['old_pred'].squeeze(0)
    similarity_map = output['similarity_map']

    img = torch.permute(denorm(img), (1, 2, 0)).unsqueeze(0)
    lbl_copy = copy.deepcopy(lbl)
    loc_pred_copy = copy.deepcopy(loc_pred)
    main_pred_copy = copy.deepcopy(main_pred)
    old_pred_copy = copy.deepcopy(old_pred)

    for k, v in inverted_mapping_dict.items():
        lbl_copy[lbl == k] = v
        loc_pred_copy[loc_pred == k] = v
        main_pred_copy[main_pred == k] = v
        old_pred_copy[old_pred == k] = v
    
    lbl = torch.from_numpy(lab2color(lbl_copy) / 255.).unsqueeze(0)
    loc_pred = torch.from_numpy(lab2color(loc_pred_copy) / 255.).unsqueeze(0)
    main_pred = torch.from_numpy(lab2color(main_pred_copy) / 255.).unsqueeze(0)
    old_pred = torch.from_numpy(lab2color(old_pred_copy) / 255.).unsqueeze(0)

    if similarity_map is not None: # for rasp
        n_sim_maps = similarity_map.shape[0]
        for i in range(n_sim_maps):
            similarity_maps.append(
                torch.permute(similarity_map[i].unsqueeze(0).repeat(3, 1, 1), (1, 2, 0)).unsqueeze(0)
            )
        similarity_maps = torch.cat(similarity_maps, dim=0)
    
    main_pred_overlayed = torch.from_numpy(
        (lab2color(main_pred_copy) / 255.) * 0.65 + img.squeeze(0).numpy() * 0.35
    ).unsqueeze(0)
    old_pred_overlayed = torch.from_numpy(
        (lab2color(old_pred_copy) / 255.) * 0.65 + img.squeeze(0).numpy() * 0.35
    ).unsqueeze(0)
    lbl_overlayed = torch.from_numpy(
        (lab2color(lbl_copy) / 255.) * 0.65 + img.squeeze(0).numpy() * 0.35
    ).unsqueeze(0)

    # save each sample and the predictions, GT
    if similarity_map is not None: # for rasp
        concat_imgs = torch.cat(
            (img, old_pred_overlayed, similarity_maps, main_pred_overlayed, lbl_overlayed), 
            dim=0
        ) # concat along width. dim=2
    else: # for wilson or other methods
        concat_imgs = torch.cat(
            (img, old_pred_overlayed, main_pred_overlayed, lbl_overlayed), 
            dim=0
        ) # concat along width. dim=2
    concat_imgs = torch.permute(concat_imgs, (0, 3, 1, 2)) # bs x c x h x w
    f_path = os.path.join(base_path, 'combined', f'{str(idx).zfill(4)}.jpg')
    if not os.path.exists(os.path.join(base_path, 'combined')):
        os.makedirs(os.path.join(base_path, 'combined')) 
    save_image(make_grid(concat_imgs, nrow=1), f_path)

    # save only the images
    f_path = os.path.join(base_path, 'input', f'{str(idx).zfill(4)}.jpg')
    if not os.path.exists(os.path.join(base_path, 'input')):
        os.makedirs(os.path.join(base_path, 'input'))
    save_image(make_grid(torch.permute(img, (0, 3, 1, 2)), nrow=1), f_path)

    # save only the predictions
    f_path = os.path.join(base_path, 'mainhead', f'{str(idx).zfill(4)}.png')
    if not os.path.exists(os.path.join(base_path, 'mainhead')):
        os.makedirs(os.path.join(base_path, 'mainhead'))
    save_image(make_grid(torch.permute(main_pred, (0, 3, 1, 2)), nrow=1), f_path)

    # save the semantic maps with vibrant colormaps
    if similarity_map is not None: # for rasp
        sem_base = os.path.join(base_path, 'semantic_maps', f'{str(idx).zfill(4)}')
        if not os.path.exists(sem_base):
            os.makedirs(sem_base)
        for i in range(n_sim_maps):
            f_path = os.path.join(sem_base, f'{str(i).zfill(2)}.png')
            sem_map = similarity_maps[i].numpy() * 255.
            imC = cv2.applyColorMap(sem_map.astype(np.uint8), cv2.COLORMAP_HOT)
            cv2.imwrite(f_path, imC)

    # save only the overlayed predictions
    f_path = os.path.join(base_path, 'pred_overlayed', f'{str(idx).zfill(4)}.png')
    if not os.path.exists(os.path.join(base_path, 'pred_overlayed')):
        os.makedirs(os.path.join(base_path, 'pred_overlayed'))
    save_image(make_grid(torch.permute(main_pred_overlayed, (0, 3, 1, 2)), nrow=1), f_path)

    # save only the overlayed old model predictions
    f_path = os.path.join(base_path, 'old_pred_overlayed', f'{str(idx).zfill(4)}.png')
    if not os.path.exists(os.path.join(base_path, 'old_pred_overlayed')):
        os.makedirs(os.path.join(base_path, 'old_pred_overlayed'))
    save_image(make_grid(torch.permute(old_pred_overlayed, (0, 3, 1, 2)), nrow=1), f_path)

    # save only the GT
    f_path = os.path.join(base_path, 'groundtruth', f'{str(idx).zfill(4)}.png')
    if not os.path.exists(os.path.join(base_path, 'groundtruth')):
        os.makedirs(os.path.join(base_path, 'groundtruth'))
    save_image(make_grid(torch.permute(lbl, (0, 3, 1, 2)), nrow=1), f_path)

    # save only the overlayed ground truth
    f_path = os.path.join(base_path, 'gt_overlayed', f'{str(idx).zfill(4)}.png')
    if not os.path.exists(os.path.join(base_path, 'gt_overlayed')):
        os.makedirs(os.path.join(base_path, 'gt_overlayed'))
    save_image(make_grid(torch.permute(lbl_overlayed, (0, 3, 1, 2)), nrow=1), f_path)


def visualize_images(dataset, ret_images, mapping_dict):
    # visualize images for step > 0
    cmap = color_map(dataset=dataset)
    lab2color = Label2Color(cmap)
    imgs = []
    lbls = []
    preds = []
    preds_old = []
    probs_old = []
    loc_preds = []
    localizer_probs = []
    targets_all = []
    fg_masks = []
    
    sim_masks1 = []
    sim_masks2 = []

    inverted_mapping_dict = {v: k for k, v in mapping_dict.items()}

    for i, (img, lbl, pred, loc_pred, pred_old, target_all, prob_old, localizer_prob, fg_mask) in enumerate(ret_images):
        img = torch.permute(denorm(img), (1, 2, 0)).unsqueeze(0)
        # transform model predictions to match the ground truth color maps
        lbl_copy = copy.deepcopy(lbl)
        pred_copy = copy.deepcopy(pred)
        pred_old_copy = copy.deepcopy(pred_old)
        loc_pred_copy = copy.deepcopy(loc_pred)
        target_all_copy = copy.deepcopy(target_all)

        for k, v in inverted_mapping_dict.items():
            lbl_copy[lbl == k] = v
            pred_copy[pred == k] = v
            pred_old_copy[pred_old == k] = v
            loc_pred_copy[loc_pred == k] = v
            target_all_copy[target_all == k] = v
        
        lbl = torch.from_numpy(lab2color(lbl_copy) / 255.).unsqueeze(0)
        pred = torch.from_numpy(lab2color(pred_copy) / 255.).unsqueeze(0)
        pred_old = torch.from_numpy(lab2color(pred_old_copy) / 255.).unsqueeze(0)
        loc_pred = torch.from_numpy(lab2color(loc_pred_copy) / 255.).unsqueeze(0)
        target_all = torch.from_numpy(lab2color(target_all_copy) / 255.).unsqueeze(0)
        prob_old = torch.permute(prob_old.unsqueeze(0).repeat(3, 1, 1), (1, 2, 0)).unsqueeze(0)
        localizer_prob = torch.permute(localizer_prob.unsqueeze(0).repeat(3, 1, 1), (1, 2, 0)).unsqueeze(0)
        fg_mask = torch.permute(fg_mask.unsqueeze(0).repeat(3, 1, 1), (1, 2, 0)).unsqueeze(0)

        imgs.append(img)
        lbls.append(lbl)
        preds.append(pred)
        preds_old.append(pred_old)
        loc_preds.append(loc_pred)
        targets_all.append(target_all)
        probs_old.append(prob_old)
        localizer_probs.append(localizer_prob)
        fg_masks.append(fg_mask)

    
    imgs = torch.cat(imgs, dim=0)
    lbls = torch.cat(lbls, dim=0)
    preds = torch.cat(preds, dim=0)
    preds_old = torch.cat(preds_old, dim=0)
    loc_preds = torch.cat(loc_preds, dim=0)
    targets_all = torch.cat(targets_all, dim=0)
    probs_old = torch.cat(probs_old, dim=0)
    localizer_probs = torch.cat(localizer_probs, dim=0)
    fg_masks = torch.cat(fg_masks, dim=0)

    concat_imgs = torch.cat((imgs, lbls, preds, loc_preds, localizer_probs, preds_old, probs_old), dim=2) # concat along width
    concat_imgs = torch.permute(concat_imgs, (0, 3, 1, 2)) # bs x c x h x w
    imgs_grid = make_grid(concat_imgs, nrow=2)
    return imgs_grid

def save_images(imgs_grid, filepath='/path/to/logs/file.png'):
    # save images only in the rank 0 process
    if distributed.get_rank() == 0:
        save_image(imgs_grid, filepath)


def convert_bn2gn(module):
    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_features = module.num_features
        num_groups = num_features//16
        mod = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2gn(child))
    del module
    return mod

def _init_dist_slurm(backend, port=None):
    """Initialize slurm distributed training environment
    Code copied from https://github.com/open-mmlab/mmcv/runner/dist_utils.py

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, the a default port ``29500`` will be used

    Args:
        backend (str): Backend of torch.distributed
        port (int, optional): Master port. Defaults to None.
    """

    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.get_output(
        f'scontrol show hostname {node_list} | head -n1'
    )
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass # use MASTER_PORT in the environment variable
    else:
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_posrt())
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    distributed.init_process_group(backend=backend)

    return proc_id, ntasks

def _is_free_port(port):
    """Checks if the port is free
    Code copied from https://github.com/open-mmlab/mmcv/runner/dist_utils.py

    Args:
        port (int): Port number
    """
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)

def _find_free_port():
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port