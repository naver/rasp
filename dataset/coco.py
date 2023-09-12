import os.path as osp
import os
import torch.utils.data as data
import numpy as np
from .dataset import IncrementalSegmentationDataset, FSSDataset
from PIL import Image
import pickle5 as pkl
from .utils import get_bert_similarity

ignore_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # starting from 1=person

COCO_CLASSES = {
    0:  'background',
    1:  'person',
    2:	'bicycle',
    3:	'car',
    4:	'motorcycle',
    5:	'airplane',
    6:	'bus',
    7:	'train',
    8:	'truck',
    9:	'boat',
    10:	'traffic light',
    11:	'fire hydrant',
    12:	'street sign',
    13:	'stop sign',
    14:	'parking meter',
    15:	'bench',
    16:	'bird',
    17:	'cat',
    18:	'dog',
    19:	'horse',
    20:	'sheep',
    21:	'cow',
    22:	'elephant',
    23:	'bear',
    24:	'zebra',
    25:	'giraffe',
    26:	'hat',
    27:	'backpack',
    28:	'umbrella',
    29:	'shoe',
    30:	'eye glasses',
    31:	'handbag',
    32:	'tie',
    33:	'suitcase',
    34:	'frisbee',
    35:	'skis',
    36:	'snowboard',
    37:	'sports ball',
    38:	'kite',
    39:	'baseball',
    40:	'baseball glove',
    41:	'skateboard',
    42:	'surfboard',
    43:	'tennis racket',
    44:	'bottle',
    45:	'plate',
    46:	'wine glass',
    47:	'cup',
    48:	'fork',
    49:	'knife',
    50:	'spoon',
    51:	'bowl',
    52:	'banana',
    53:	'apple',
    54:	'sandwich',
    55:	'orange',
    56:	'broccoli',
    57:	'carrot',
    58:	'hot dog',
    59:	'pizza',
    60:	'donut',
    61:	'cake',
    62:	'chair',
    63:	'couch',
    64:	'potted plant',
    65:	'bed',
    66:	'mirror',
    67:	'dining table',
    68:	'window',
    69:	'desk',
    70:	'toilet',
    71:	'door',
    72:	'tv',
    73:	'laptop',
    74:	'mouse',
    75:	'remote',
    76:	'keyboard',
    77:	'cell phone',
    78:	'microwave',
    79:	'oven',
    80:	'toaster',
    81:	'sink',
    82:	'refrigerator',
    83:	'blender',
    84:	'book',
    85:	'clock',
    86:	'vase',
    87:	'scissors',
    88:	'teddy bear',
    89:	'hair drier',
    90:	'toothbrush',
    91:	'hair brush'
}

# glove does not support all the class names in COCO
# hence they have been replaced with most meaningful synonyms
COCO_GLOVE_CLASSES = {
    0:  'unknown',
    1:  'person',
    2:	'bicycle',
    3:	'car',
    4:	'motorcycle',
    5:	'airplane',
    6:	'bus',
    7:	'train',
    8:	'truck',
    9:	'boat',
    10:	'semaphore',        # original: traffic light
    11:	'hydrant',          # original: fire hydrant
    12:	'placard',          # original: street sign
    13:	'signposts',        # original: stop sign
    14:	'timer',            # original: parking meter
    15:	'bench',
    16:	'bird',
    17:	'cat',
    18:	'dog',
    19:	'horse',
    20:	'sheep',
    21:	'cow',
    22:	'elephant',
    23:	'bear',
    24:	'zebra',
    25:	'giraffe',
    26:	'hat',
    27:	'backpack',
    28:	'umbrella',
    29:	'shoe',
    30:	'eyeglasses',       # original: eye glasses
    31:	'handbag',
    32:	'necktie',          # original: tie
    33:	'suitcase',
    34:	'frisbee',
    35:	'skis',
    36:	'snowboard',
    37:	'balls',            # original: sports ball
    38:	'kite',
    39:	'baseball',
    40:	'glove',            # orignal: baseball glove
    41:	'skateboard',
    42:	'surfboard',
    43:	'racket',           # original: tennis racket
    44:	'bottle',
    45:	'plate',
    46:	'wineglass',        # original: wine glass
    47:	'cup',
    48:	'fork',
    49:	'knife',
    50:	'spoon',
    51:	'bowl',
    52:	'banana',
    53:	'apple',
    54:	'sandwich',
    55:	'orange',
    56:	'broccoli',
    57:	'carrot',
    58:	'hotdog',           # original: hot dog
    59:	'pizza',
    60:	'donut',
    61:	'cake',
    62:	'chair',
    63:	'couch',
    64:	'potted',          # original: potted plant
    65:	'bed',
    66:	'mirror',
    67:	'tables',          # original: dining table
    68:	'window',
    69:	'desk',
    70:	'toilet',
    71:	'door',
    72:	'tv',
    73:	'laptop',
    74:	'mouse',
    75:	'remote',
    76:	'keyboard',
    77:	'cellphone',        # original: cell phone
    78:	'microwave',
    79:	'oven',
    80:	'toaster',
    81:	'sink',
    82:	'refrigerator',
    83:	'blender',
    84:	'book',
    85:	'clock',
    86:	'vase',
    87:	'scissors',
    88:	'teddy',            # original: teddy bear
    89:	'drier',            # original: hair drier
    90:	'toothbrush',
    91:	'hairbrush'         # original: hair brush
}

classes = {
    0: 'background',
    8: 'truck',
    10:	'traffic light',
    11:	'fire hydrant',
    13:	'stop sign',
    14:	'parking meter',
    15:	'bench',
    22:	'elephant',
    23:	'bear',
    24:	'zebra',
    25:	'giraffe',
    27:	'backpack',
    28:	'umbrella',
    31:	'handbag',
    32:	'tie',
    33:	'suitcase',
    34:	'frisbee',
    35:	'skis',
    36:	'snowboard',
    37:	'sports ball',
    38:	'kite',
    39:	'baseball bat',
    40:	'baseball glove',
    41:	'skateboard',
    42:	'surfboard',
    43:	'tennis racket',
    46:	'wine glass',
    47:	'cup',
    48:	'fork',
    49:	'knife',
    50:	'spoon',
    51:	'bowl',
    52:	'banana',
    53:	'apple',
    54:	'sandwich',
    55:	'orange',
    56:	'broccoli',
    57:	'carrot',
    58:	'hot dog',
    59:	'pizza',
    60:	'donut',
    61:	'cake',
    65:	'bed',
    70:	'toilet',
    73:	'laptop',
    74:	'mouse',
    75:	'remote',
    76:	'keyboard',
    77:	'cell phone',
    78:	'microwave',
    79:	'oven',
    80:	'toaster',
    81:	'sink',
    82:	'refrigerator',
    84:	'book',
    85:	'clock',
    86:	'vase',
    87:	'scissors',
    88:	'teddy bear',
    89:	'hair drier',
    90:	'toothbrush'
}


class COCO(data.Dataset):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 indices=None):

        root = osp.expanduser(root)
        base_dir = "coco"
        ds_root = osp.join(root, base_dir)
        splits_dir = osp.join(ds_root, 'split')

        if train:
            self.image_set = "train"
            split_f = osp.join(splits_dir, 'train.txt')
            folder = 'train2017'
        else:
            self.image_set = "val"
            split_f = osp.join(splits_dir, 'val.txt')
            folder = 'val2017'

        ann_folder = "annotations"

        with open(osp.join(split_f), "r") as f:
            files = f.readlines()

        self.images = [(osp.join(ds_root, "images", folder, x[:-1] + ".jpg"),
                        osp.join(ds_root, ann_folder, folder, x[:-1] + ".png")) for x in files]

        self.img_lvl_labels = np.load(osp.join(ds_root, f"1h_labels_{self.image_set}.npy"))

        self.transform = transform
        self.indices = indices if indices is not None else np.arange(len(self.images))
        # self.img_lvl_only = False

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

class COCOFSSegmentation(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, stuff=False):
        root = os.path.expanduser(root)
        base_dir = 'coco' if not stuff else 'coco_stuff'
        ds_root = os.path.join(root, base_dir)
        splits_dir = os.path.join(ds_root, 'split')
        self.transform = transform
        self.target_transform = target_transform

        if train:
            split_f = os.path.join(splits_dir, 'train.txt')
            folder = 'train2017'
        else:
            split_f = os.path.join(splits_dir, 'val.txt')
            folder = 'val2017'
        
        img_folder = 'images'
        ann_folder = 'annotations'

        with open(split_f) as f:
            files = f.readlines()
        
        if train:
            path = '/inverse_dict_train_coco.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))
        else:
            path = '/inverse_dict_test_coco.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))
        
        self.images = [(os.path.join(ds_root, img_folder, folder, x[:-1] + '.jpg'),
                        os.path.join(ds_root, ann_folder, folder, x[:-1] + '.png')) for x in files]
        if train:
            self.img_lvl_labels = np.load(os.path.join(splits_dir, 'coco_fss_1h_labels_train.npy'))
        else:
            self.img_lvl_labels = np.load(os.path.join(splits_dir, 'coco_fss_1h_labels_val.npy'))
        
        assert len(self.img_lvl_labels) == len(self.images), "Number of images and one hot labels annotations do not match. Something is wrong!"
    
    @property
    def class_to_images(self):
        return self.class_to_images_
    
    def __getitem__(self, index):
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        img_lvl_label = self.img_lvl_labels[index]

        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target, img_lvl_label
    
    def __len__(self):
        return len(self.images)


class COCOIncremental(IncrementalSegmentationDataset):
    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        full_voc = COCO(root, train, transform=None, indices=indices)
        return full_voc
    
    def class_dict(self):
        """
        Returns the class names in the PASCAL VOC data set
        """
        return classes
    
    def imagenet_wnid_dict(self):
        """
        Returns the imagenet and PASCAL VOC corresponding class dictionary
        """
        return None

    def class_affinity(self, scaling):
        """
        Returns the affinity between a given pair of classes
        """
        # for binary tree based similarity
        #return affinity
        # for glove based similarity
        return None

class COCOFSSegmentationIncremental(FSSDataset):
    def make_dataset(self, root, train):
        full_coco = COCOFSSegmentation(root, train, transform=None)
        return full_coco
    
    def class_dict(self, similarity_type='transformer'):
        """
        Returns the class names in the PASCAL VOC data set
        """
        classes_dict = {
            'transformer': COCO_CLASSES, 
            'glove': COCO_GLOVE_CLASSES,
            'tree': COCO_CLASSES
        }
        return classes_dict[similarity_type]
    
    def class_affinity(self, scaling=5, similarity_type='transformer'):
        """
        Returns the affinity between a given pair of classes
        """
        if similarity_type == 'transformer':
            sim = get_bert_similarity(self.class_dict(similarity_type), dim=384, scaling=scaling)
        elif similarity_type == 'glove':
            sim = get_glove_similarity(self.class_dict(similarity_type), dim=100, scaling=scaling)
        else:
            raise NotImplementedError(f'COCO does not support {similarity_type} semantic similarity')
        return sim
    
    def get_order(self):
        # Returns in order all the classes seen so far
        return self.order