import os
import numpy as np
from PIL import Image

n_classes = 91 # 91 classes in COCO excluding the bkg class
# the classes that are ignored for being noisy, etc
ignore_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]
coco_root = 'data/coco/'
split = 'train' # train/val
splits_dir = os.path.join(coco_root, 'split/')
split_f = os.path.join(splits_dir, split + '.txt')

annotations_dir = os.path.join(coco_root, 'annotations')
save_dir = splits_dir

with open(split_f) as f:
    files = f.readlines()

one_hot_labels = np.zeros((len(files), n_classes))

for i, x in enumerate(files):
    ann_f = os.path.join(annotations_dir, split + '2017', x[:-1] + '.png')
    labels = np.asarray(Image.open(ann_f)).flatten()
    unique_lbls = list(np.unique(labels))
    # remove labels [0, 255] + ignore_labels
    remove_labels = [0, 255] + ignore_labels
    for remove_lbl in remove_labels:
        if remove_lbl in unique_lbls:
            remove_idx = unique_lbls.index(remove_lbl)
            del unique_lbls[remove_idx]
    
    if len(unique_lbls) == 0:
        continue
    else:
        unique_lbls = np.asarray(unique_lbls)
        temp_hot = np.zeros((len(unique_lbls), n_classes))
        temp_hot[np.arange(unique_lbls.size), unique_lbls - 1] = 1
        one_hot_labels[i] = np.asarray(temp_hot.sum(axis=0))
    
# save the one hot numpy file
save_path = os.path.join(save_dir, 'coco_fss_1h_labels_' + split + '.npy')
print(f'Saving file at {save_path}')
np.save(save_path, one_hot_labels)