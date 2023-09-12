import os
import numpy as np
from PIL import Image

n_classes = 200 # 200 classes in CUB excluding the bkg class
cub_root = 'data/cub/'
split = 'val' # train/val
splits_dir = os.path.join(cub_root, 'split/')
split_f = os.path.join(splits_dir, split + '.txt')

annotations_dir = os.path.join(cub_root, 'annotations')
save_dir = splits_dir

with open(split_f) as f:
    files = f.readlines()

one_hot_labels = np.zeros((len(files), n_classes))

for i, x in enumerate(files):
    ann_f = os.path.join(annotations_dir, x[:-1] + '.png')
    labels = np.asarray(Image.open(ann_f)).flatten()
    unique_lbls = list(np.unique(labels))

    remove_labels = [0, 255]
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
save_path = os.path.join(save_dir, 'cub_fss_1h_labels_' + split + '.npy')
print(f'Saving file at {save_path}')
np.save(save_path, one_hot_labels)