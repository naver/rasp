import torch
import numpy as np
import bisect
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from torch import distributed
import gensim.downloader


def image_labels(dataset):
    images = []
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        images.append(cls)
    return images


def group_images(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if all(x in labels_cum for x in cls):
            for x in cls:
                if x in labels:
                    idxs[x].append(i)
    return idxs


def group_images_bkg(dataset, labels):
    # Group images based on the label in LABELS (using labels not reordered)
    idxs = {lab: [] for lab in labels}

    labels_cum = labels + [0, 255]
    for i in range(len(dataset)):
        cls, count_classes = np.unique(np.array(dataset[i][1]), return_counts=True)
        count = 0
        if all(x in labels_cum for x in cls):
            for j, cl in enumerate(cls):
                if cl == 0 or cl == 255:
                    count += count_classes[j]
            for x in cls:
                if x in labels:
                    idxs[x].append((i, count))
    return idxs

def filter_images(dataset, labels, labels_old=None):
    idxs = []
    if 0 in labels:
        labels.remove(0)

    if distributed.get_rank() == 0:
        print('Filtering images...')
    if labels_old is None:
        labels_old = []
    labels_cum = labels_old + labels + [0, 255]

    fil = lambda c: any(x in labels for x in c) and all(x in labels_cum for x in c)

    for i in range(len(dataset)):
        cls = np.unique(np.array(dataset[i][1]))
        if fil(cls):
            idxs.append(i)
        if i % 1000 == 0:
            if distributed.get_rank() == 0:
                print(f'\t{i}/{len(dataset)}...')
    return idxs


class Subset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (callable): way to transform the images and the targets
        target_transform(callable): way to transform the target labels
    """

    def __init__(self, dataset, indices, transform=None, target_transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target, l1h = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample, target = self.transform(sample, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, l1h

    def __len__(self):
        return len(self.indices)


class ConcatDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class MaskLabels:
    """
    Use this class to mask labels that you don't want in your dataset.
    Arguments:
    labels_to_keep (list): The list of labels to keep in the target images
    mask_value (int): The value to replace ignored values (def: 0)
    """
    def __init__(self, labels_to_keep, mask_value=0):
        self.labels = labels_to_keep
        self.value = torch.tensor(mask_value, dtype=torch.uint8)

    def __call__(self, sample):
        # sample must be a tensor
        assert isinstance(sample, torch.Tensor), "Sample must be a tensor"

        sample.apply_(lambda t: t.apply_(lambda x: x if x in self.labels else self.value))

        return sample

def get_bert_similarity(class_map, dim=768, scaling=5):
    # to get the semantic similarity between two classes using a sentence transformer
    embeddings = np.zeros((len(class_map), dim))
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # extract sentence embeddings corresponding to the [CLS] token
    #for i in range(len(classes)):
    cls_idx = 0
    for k, v in class_map.items():
        #label_name = classes[i] if classes[i] != 'background' else 'unknown'
        label_name = v if v != 'background' else 'unknown'
        prompt = f'this is an image of {label_name}'
        embeddings[cls_idx] = np.asarray(model.encode(prompt))
        cls_idx += 1
    
    bert_affinity = np.zeros((len(class_map), len(class_map)))
    for i in range(len(class_map)):
        for j in range(len(class_map)):
            # compute cosine_distance = 1 - cosine similarity
            bert_affinity[i][j] = cosine(embeddings[i], embeddings[j])
    bert_affinity = np.exp(-scaling * bert_affinity)
    return bert_affinity

def get_glove_similarity(class_map, dim=100, scaling=5):
    # to get the semantic similarity between two classes using a glove embeddings
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
    embeddings = np.zeros((len(class_map), dim))

    for k, v in class_map.items():
        embeddings[k] = glove_vectors[v]
    
    glove_affinity = np.zeros((len(class_map), len(class_map)))
    for i in range(len(class_map)):
        for j in range(len(class_map)):
            # compute cosine_distance = 1 - cosine similarity
            glove_affinity[i][j] = cosine(embeddings[i], embeddings[j])
    
    glove_affinity = np.exp(-5 * glove_affinity)
    return glove_affinity

def get_tree_similarity(class_map=None, dim=0, scaling=2):
    # computed from wordnet-inspired binary tree
    # an ij entry in the matrix represents the number of nodes/hops
    # between two nodes (or classes) i and j in the tree
    # background class is considered as the root node of the tree
    # currently supported for Pascal VOC only
    tree_affinity = np.exp(-np.asarray(
    [
        [0, 3, 7, 3, 4,	4, 6, 6, 6,	5, 7, 4, 6,	7,	7, 4, 2, 6,	5,	6,	4],
        [3, 0,	6, 6, 3, 5,	5, 5, 9, 6,	10,	5, 9, 10, 6, 7,	5, 9, 6, 5,	5],
        [7, 6, 0, 10, 5, 9, 5,	3, 13, 10, 14, 9, 13, 14, 2, 11, 9,	13,	10,	5, 9],
        [3, 6, 10, 0, 7, 7, 9,	9, 5, 8, 6,	7, 5, 6, 10, 3,	3, 5, 8, 9,	7],
        [4, 3, 5, 7, 0, 6,	4, 4, 10, 7, 11, 6,	10,	11,	5,	8,	6,	10,	7,	4,	6],
        [4, 5, 9, 7, 6, 0, 8, 8, 10, 5, 11, 4, 10,	11,	9, 8, 6, 10, 5,	8, 2],
        [6, 5,	5,	9,	4,	8,	0,	4,	12,	9,	13,	8,	12,	13,	5,	10,	8,	12,	9,	2,	8],
        [6, 5,	3,	9,	4,	8,	4,	0,	12,	9,	13,	8,	12,	13,	3,	10,	8,	12,	9,	4,	8],
        [6, 9,	13,	5,	10,	10,	12,	12,	0,	11,	5,	10,	2,	5,	13,	4,	6,	4,	11,	12,	10],
        [5, 6,	10,	8,	7,	5,	9,	9,	11,	0,	12,	3,	11,	12,	10,	9,	7,	11,	2,	9,	5],
        [7, 10,	14,	6,	11,	11,	13,	13,	5,	12,	0,	11,	5,	2,	14,	5,	7,	3,	12,	13,	11],
        [4, 5,	9,	7,	6,	4,	8,	8,	10,	3,	11,	0,	10,	11,	9,	8,	6,	10,	3,	9,	4],
        [6, 9,	13,	5,	10,	10,	12,	12,	2,	11,	5,	10,	0,	5,	13,	4,	6,	4,	11,	12,	10],
        [7, 10,	14,	6,	11,	11,	13,	13,	5,	12,	2,	11,	5,	0,	14,	5,	7,	3,	12,	13,	11],
        [7, 6,	2,	10,	5,	9,	5,	3,	13,	10,	14,	9,	13,	14,	0,	11,	9,	13,	10,	5,	9],
        [4, 7,	11,	3,	8,	8,	10,	10,	4,	9,	5,	8,	4,	5,	11,	0,	4,	4,	9,	10,	8],
        [2, 5,	9,	3,	6,	6,	8,	8,	6,	7,	7,	6,	6,	7,	9,	4,	0,	6,	7,	8,	6],
        [6, 9,	13,	5,	10,	10,	12,	12,	4,	11,	3,	10,	4,	3,	13,	4,	6,	0,	11,	12,	10],
        [5, 6,	10,	8,	7,	5,	9,	9,	11,	2,	12,	3,	11,	12,	10,	9,	7,	11,	0,	9,	5],
        [6, 5,	5,	9,	4,	8,	2,	4,	12,	9,	13,	9,	12,	13,	5,	10,	8,	12,	9,	0,	8],
        [4, 5,	9,	7,	6,	2,	8,	8,	10,	5,	11,	4,	10,	11,	9,	8,	6,	10,	5,	8,	0]
    ])/scaling)

    return tree_affinity