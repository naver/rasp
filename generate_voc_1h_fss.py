import os
import numpy as np
from PIL import Image

n_classes = 20 # 20 classes in VOC excluding the bkg class
voc_root = 'data/voc/'
split = 'test' # train/test
splits_dir = os.path.join(voc_root, 'split_fss/')
indices_fname = os.path.join(splits_dir, split + '_ids.npy')
annotations_dir = os.path.join(voc_root, 'SegmentationClassAug/')
indices_arr = np.load(indices_fname)
save_dir = 'data/voc/split_fss'

one_hot_labels = np.zeros((len(indices_arr), n_classes))

for i in range(len(indices_arr)):
    annotation_fname = os.path.join(annotations_dir, indices_arr[i] + '.png')
    labels = np.asarray(Image.open(annotation_fname)).flatten()
    unique_lbls = list(np.unique(labels))
    # remove labels [0, 255]
    for remove_lbl in [0, 255]:
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
# rename test with val in case the split is test
np.save(os.path.join(save_dir, 'voc_fss_1h_labels_' + split + '.npy'), one_hot_labels)


