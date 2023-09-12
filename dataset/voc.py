import os
import torch.utils.data as data
from .dataset import IncrementalSegmentationDataset, FSSDataset
import numpy as np
from scipy.spatial.distance import cosine
import gensim.downloader
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from .utils import get_bert_similarity, get_glove_similarity, get_tree_similarity
from .coco import COCO_CLASSES, COCO_GLOVE_CLASSES

from PIL import Image
import pickle5 as pkl

from torch import distributed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

glove_classes = {
    0: 'unknown', 
    1: 'plane', 
    2: 'bicycle', 
    3: 'bird', 
    4: 'boat', 
    5: 'bottle', 
    6: 'bus', 
    7: 'car', 
    8: 'cat', 
    9: 'chair', 
    10: 'cow', 
    11: 'tables', 
    12: 'dog', 
    13: 'horse', 
    14: 'motorbike', 
    15: 'person', 
    16: 'potted', 
    17: 'sheep', 
    18: 'sofa', 
    19: 'train', 
    20: 'tv'
}

coco_voc_classes = {
    0: 'unknown', 
    1:	'truck',
    2:	'traffic light',
    3:	'fire hydrant',
    4:	'stop sign',
    5:	'parking meter',
    6:	'bench',
    7:	'elephant',
    8:	'bear',
    9:	'zebra',
    10:	'giraffe',
    11:	'backpack',
    12:	'umbrella',
    13:	'handbag',
    14:	'tie',
    15:	'suitcase',
    16:	'frisbee',
    17:	'skis',
    18:	'snowboard',
    19:	'sports ball',
    20:	'kite',
    21:	'baseball bat',
    22:	'baseball glove',
    23:	'skateboard',
    24:	'surfboard',
    25:	'tennis racket',
    26:	'wine glass',
    27:	'cup',
    28:	'fork',
    29:	'knife',
    30:	'spoon',
    31:	'bowl',
    32:	'banana',
    33:	'apple',
    34:	'sandwich',
    35:	'orange',
    36:	'broccoli',
    37:	'carrot',
    38:	'hot dog',
    39:	'pizza',
    40:	'donut',
    41:	'cake',
    42:	'bed',
    43:	'toilet',
    44:	'laptop',
    45:	'mouse',
    46:	'remote',
    47:	'keyboard',
    48:	'cell phone',
    49:	'microwave',
    50:	'oven',
    51:	'toaster',
    52:	'sink',
    53:	'refrigerator',
    54:	'book',
    55:	'clock',
    56:	'vase',
    57:	'scissors',
    58:	'teddy bear',
    59:	'hair drier',
    60:	'toothbrush',
    61:	'person',
    62:	'bicycle',
    63:	'car',
    64:	'motorcycle',
    65:	'airplane',
    66:	'bus',
    67:	'train',
    68:	'boat',
    69: 'bird',
    70: 'cat',
    71: 'dog',
    72: 'horse',
    73: 'sheep',
    74: 'cow',
    75: 'bottle',
    76: 'chair',
    77: 'sofa',
    78: 'potted plant',
    79: 'dinning table',
    80: 'tv'
}

imagenet_map = {
	1:	['n04552348', 'n02690373'], # 'n04266014', 'n02692877'],
	2:	['n02835271', 'n03792782'], # 'n04509417'
	3:	['n02002556', 'n01532829', 'n02058221', 'n01531178'], # 'n01601694', 'n01560419', 'n01558993', 'n01582220'],
	4:	['n04273569', 'n03662601', 'n04612504', 'n02981792', 'n04483307'], # 'n03447447', 'n03344393' , 'n03095699', 'n03947888'
	5: 	['n02823428', 'n03983396', 'n04591713'], # 'n04557648'
	6: 	['n04487081', 'n03769881', 'n04146614'],
	7: 	['n02930766', 'n04037443', 'n04285008'], # 'n03594945', 'n03770679', 'n02814533'
	8:	['n02124075', 'n02123394', 'n02123597'], # 'n02123045', 'n02123159'
	9:	['n03376595', 'n04099969'], # 'n02791124'
	10: ['n02403003'],
	11: ['n03201208'],
	12:	['n02096294', 'n02110341', 'n02093256', 'n02106382', 'n02113712'],
        # 'n02094433', 'n02111277'],
        # 'n02110806', 'n02107142', 'n02100583', 'n02099601', 'n02106166', 'n02109961', 'n02100877', 
        # 'n02106662', 'n02098105'],
	13: ['n02389026', 'n03538406'],
	14:	['n03785016', 'n03791053'],
	15:	['n09835506', 'n10148035'], # 'n10565667', 'n04456115'
	16:	['n03991062'],
	17: ['n02415577', 'n02412080'],
	18:	['n04344873'],
	19:	['n02917067', 'n04310018', 'n03272562'],
	20: ['n03782006']
}

task_list = ['person', 'animals', 'vehicles', 'indoor']
tasks = {
    'person': [15],
    'animals': [3, 8, 10, 12, 13, 17],
    'vehicles': [1, 2, 4, 6, 7, 14, 19],
    'indoor': [5, 9, 11, 16, 18, 20]
}

coco_map = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None,
                 as_coco=False,
                 saliency=False,
                 pseudo=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = 'train' if train else 'val'
        base_dir = "voc"
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, 'splits')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Download it')

        mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
        assert os.path.exists(mask_dir), "SegmentationClassAug not found"

        if as_coco:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug_ascoco.txt')
            else:
                split_f = os.path.join(splits_dir, 'val_ascoco.txt')
        else:
            if train:
                split_f = os.path.join(splits_dir, 'train_aug.txt')
            else:
                split_f = os.path.join(splits_dir, 'val.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]
        if saliency:
            self.saliency_images = [x[0].replace("JPEGImages", "SALImages")[:-3] + "png" for x in self.images]
        else:
            self.saliency_images = None

        if pseudo is not None and train:
            if not as_coco:
                self.images = [(x[0], x[1].replace("SegmentationClassAug", f"PseudoLabels/{pseudo}/rw/")) for x in self.images]
            else:
                self.images = [(x[0], x[1].replace("SegmentationClassAugAsCoco", f"PseudoLabels/{pseudo}/rw")) for x in
                               self.images]
        if as_coco:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"cocovoc_1h_labels_{self.image_set}.npy"))
        else:
            self.img_lvl_labels = np.load(os.path.join(voc_root, f"voc_1h_labels_{self.image_set}.npy"))

        self.indices = indices if indices is not None else np.arange(len(self.images))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[self.indices[index]][0]).convert('RGB')
        target = Image.open(self.images[self.indices[index]][1])
        img_lvl_lbls = self.img_lvl_labels[self.indices[index]]

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target, img_lvl_lbls

    def __len__(self):
        return len(self.indices)


class VOCFSSSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        train (bool): Use train (True) or test (False) split
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    
    def __init__(self, root='data', train=True, transform=None, coco_labels=False):

        if train:
            split = 'train_ids'
        else:
            split = 'test_ids'
        
        self.root = os.path.expanduser(root)
        self.transform = transform

        voc_root = os.path.join(self.root, 'voc/')
        splits_dir = os.path.join(self.root, 'voc/split_fss/')

        if not os.path.isdir(voc_root):
            raise RuntimeError(f'Dataset not found in {voc_root}.' +
                               f'Download it with download_voc.sh and link it to the {voc_root} folder.')
        
        if train:
            self.class_to_images_ = pkl.load(open(splits_dir + 'inverse_dict_train.pkl', 'rb'))
        else:
            self.class_to_images_ = None
        
        self.images = np.load(os.path.join(splits_dir, split + '.npy'))
        if coco_labels:
            annotation_folder = "annotations_coco"
        else:
            annotation_folder = "SegmentationClassAug"
        
        self.images = [(os.path.join(voc_root, "JPEGImages", i + '.jpg'), os.path.join(voc_root, annotation_folder, i + '.png')) 
                       for i in self.images]
        if train:
            self.img_lvl_labels = np.load(os.path.join(splits_dir, 'voc_fss_1h_labels_train.npy'))
        else:
            self.img_lvl_labels = np.load(os.path.join(splits_dir, 'voc_fss_1h_labels_val.npy'))
        
        assert len(self.img_lvl_labels) == len(self.images), "Number of images and one hot labels annotations do not match. Something is wrong!"
    
    @property
    def class_to_images(self):
        return self.class_to_images_
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        img_lvl_label = self.img_lvl_labels[index]
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target, img_lvl_label
    
    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, saliency=saliency, pseudo=pseudo)
        return full_voc
    
    def class_dict(self, similarity_type='transformer'):
        """
        Returns the class names in the PASCAL VOC data set
        """
        classes_dict = {
            'transformer': classes, 
            'glove': glove_classes,
            'tree': classes
        }
        return classes_dict[similarity_type]
    
    def imagenet_wnid_dict(self):
        """
        Returns the imagenet and PASCAL VOC corresponding class dictionary
        """
        return imagenet_map

    def class_affinity(self, scaling, similarity_type='transformer'):
        """
        Returns the affinity between a given pair of classes
        """
        if similarity_type == 'transformer':
            sim = get_bert_similarity(self.class_dict(similarity_type), dim=384, scaling=scaling)
        elif similarity_type == 'glove':
            sim = get_glove_similarity(self.class_dict(similarity_type), dim=100, scaling=scaling)
        else:
            sim = get_tree_similarity(scaling=scaling)
        return sim
    
    def get_order(self):
        # Returns in order all the classes seen so far
        return self.order


class VOCasCOCOSegmentationIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = VOCSegmentation(root, train, transform=None, indices=indices, as_coco=True,
                                   saliency=saliency, pseudo=pseudo)
        return full_voc
    
    def class_dict(self, similarity_type='transformer'):
        """
        Returns the class names in the data set
        """
        classes_dict = {
            'transformer': COCO_CLASSES, 
            'glove': COCO_GLOVE_CLASSES,
            'tree': COCO_CLASSES
        }
        return classes_dict[similarity_type]
    
    def imagenet_wnid_dict(self):
        """
        Returns the imagenet and PASCAL VOC corresponding class dictionary
        """
        # not supported
        return None

    def class_affinity(self, scaling, similarity_type='transformer'):
        """
        Returns the affinity between a given pair of classes
        """
        if similarity_type == 'transformer':
            sim = get_bert_similarity(self.class_dict(similarity_type), dim=384, scaling=scaling)
        elif similarity_type == 'glove':
            sim = get_glove_similarity(self.class_dict(similarity_type), dim=100, scaling=scaling)
        else:
            raise NotImplementedError(f'COCO-to-VOC does not support {similarity_type} semantic similarity')
        return sim
    
    def get_order(self):
        # Returns in order all the classes seen so far
        return self.order

class VOCFSSSegmentationIncremental(FSSDataset):
    def make_dataset(self, root, train):
        full_voc = VOCFSSSegmentation(root, train, transform=None)
        return full_voc
    
    def class_dict(self, similarity_type='transformer'):
        """
        Returns the class names in the PASCAL VOC data set
        """
        classes_dict = {
            'transformer': classes, 
            'glove': glove_classes,
            'tree': classes
        }
        return classes_dict[similarity_type]
    
    def class_affinity(self, scaling, similarity_type='transformer'):
        """
        Returns the affinity between a given pair of classes
        """
        if similarity_type == 'transformer':
            sim = get_bert_similarity(self.class_dict(similarity_type), dim=384, scaling=scaling)
        elif similarity_type == 'glove':
            sim = get_glove_similarity(self.class_dict(similarity_type), dim=100, scaling=scaling)
        else:
            sim = get_tree_similarity(scaling=scaling)
        return sim
    
    def get_order(self):
        # Returns in order all the classes seen so far
        return self.order


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])
        