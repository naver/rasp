import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import pickle

# TODO: anonymize paths
root_dir = '/scratch/1/user/sroy/ciss/birds/CUB_200_2011'
images_dir = '/scratch/1/user/sroy/ciss/birds/CUB_200_2011/images'
ann_dir = '/scratch/1/user/sroy/ciss/birds/CUB_200_2011/segmentations'

# preprocessed file dirs
dest_img_dir = '/scratch/1/user/sroy/ciss/birds/CUB_200_2011/bird_images'
dest_ann_dir = '/scratch/1/user/sroy/ciss/birds/CUB_200_2011/bird_segmentations'

os.makedirs(dest_img_dir, exist_ok=True)
os.makedirs(dest_ann_dir, exist_ok=True)
os.makedirs(os.path.join(root_dir, 'split'), exist_ok=True)

class_dict = {0: 'unknown'}
train_list = []
val_list = []
val_frac = 0.3

for _, dirs, _ in os.walk(images_dir):
    dirs.sort()
    for name in dirs:
        class_id = int(name.split('.')[0])
        bird_species = name.split('.')[1].split('_')
        bird_species = ' '.join(bird_species)
        class_dict.update({class_id: bird_species})

        files = os.listdir(os.path.join(images_dir, name))
        for f in tqdm(files, desc=name):
            # move the images to a destination folder
            img_save_path = os.path.join(dest_img_dir, f)
            shutil.copy(
                os.path.join(images_dir, name, f),
                img_save_path
            )

            # transform the annotations and move them as well
            seg_map = np.asarray(Image.open(
                os.path.join(ann_dir, name, f.split('.')[0] + '.png')
            ).convert('L'))

            max_val = np.amax(seg_map)
            fg_mask = seg_map > int(0.9 * max_val) # foreground pixels
            temp_map = np.zeros((seg_map.shape))
            temp_map[fg_mask] = class_id
            PIL_map = Image.fromarray(np.uint8(temp_map)).convert('L')
            ann_save_path = os.path.join(dest_ann_dir, f.split('.')[0] + '.png')
            PIL_map.save(ann_save_path)

            # check if both the image and annotation files have been successfully copies
            # to the destination directories
            isCopied = os.path.exists(img_save_path) and os.path.exists(ann_save_path)
            assert isCopied, f'Either of ({img_save_path}, {ann_save_path}) or both could not be copied!'

        files = [f.split('.')[0] for f in files]

        val_list.extend(files[:int(val_frac * len(files))])
        train_list.extend(files[int(val_frac * len(files)):])

        assert len(files) == (len(files[int(val_frac * len(files)):]) + len(files[:int(val_frac * len(files))])),\
         f"For {name} the lengths do not match {len(files)} != {(len(files[int(val_frac * len(files)):]) + len(files[:int(val_frac * len(files))]))}"

# dump the train and val list
val_list = '\n'.join(val_list) + '\n'
train_list = '\n'.join(train_list) + '\n'

with open(os.path.join(os.path.join(root_dir, 'split', 'val.txt')), 'w') as f:
    f.write(val_list)

with open(os.path.join(os.path.join(root_dir, 'split', 'train.txt')), 'w') as f:
    f.write(train_list)

f.close()

with open(os.path.join(root_dir, 'split', 'classes.pkl'), 'wb') as handle:
    pickle.dump(class_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

handle.close()