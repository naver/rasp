import os
import torch
import torch.utils.data as data
from torch import from_numpy
import numpy as np
import random
from torch import distributed
from torchvision import datasets
from PIL import Image
import torch.nn.functional as F

from .utils import Subset, ConcatDataset, filter_images

FILTER_FSL = False

class IncrementalSegmentationDataset(data.Dataset):
    def __init__(self,
                 root,
                 step_dict,
                 train=True,
                 transform=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 masking_value=0,
                 step=0,
                 weakly=False,
                 saliency=False,
                 pseudo=None,
                 replay=False,
                 replay_size=0.):
        
        if train:
            idxs = list()
            prev_idxs = list()
            if idxs_path is not None:
                for idx_path in idxs_path:
                    if os.path.exists(idx_path):
                        data_idxs = np.load(idx_path)
                        idx_step = idx_path.split('/')[-1].split('.')[0].split('-')[-1]
                        if replay and (int(idx_step) < step) and step > 0:
                            
                            # check if the previous-idxs are already saved in the memory or not
                            mem_fname = os.path.join("/".join(idx_path.split('/')[:-1]), f"train-mem-{idx_step}.npy")
                            # if exists then load from the saved npy files
                            if os.path.exists(mem_fname):
                                data_idxs = np.load(mem_fname)
                            else: # if not then save a npy file
                                data_idxs = random.sample(list(data_idxs), replay_size) # randomly sample
                                if distributed.get_rank() == 0:
                                    np.save(mem_fname, np.array(data_idxs, dtype=int))

                            data_idxs = data_idxs[:replay_size // step]
                            prev_idxs.append(data_idxs)
                        
                        if int(idx_step) == step:
                            idxs.append(data_idxs)
                
                if len(prev_idxs) > 0:
                    prev_idxs = np.concatenate(prev_idxs, 0) # accumulate all the previous indices
                    idxs = np.concatenate(idxs, 0)
                    idxs = np.concatenate([idxs, prev_idxs], 0) # join the current and past indices
                else:
                    idxs = np.concatenate(idxs, 0)
        else: # In both test and validation we want to use all data available (even if some images are all bkg)
            idxs = None


        self.dataset = self.make_dataset(root, train, indices=idxs, saliency=saliency, pseudo=pseudo)
        self.classes = self.class_dict()
        self.imagenet_classes = self.imagenet_wnid_dict()
        self.transform = transform
        self.weakly = weakly 
        self.saliency = saliency
        self.train = train

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step

        # it enumerates all the classes sequentially for all the tasks observed so far
        self.order = [c for s in sorted(step_dict) for c in step_dict[s]]
        
        # assert not any(l in labels_old for l in labels), "Labels and labels_old must be disjoint sets"
        # in each step there is the bkg class 0 + whatever other classes are present at that step
        if step > 0:
            # for steps greater than 0 manually include the bkg class
            self.labels = [self.order[0]] + list(step_dict[step])
        else:
            self.labels = list(step_dict[step])
        
        self.labels_old = [lbl for s in range(step) for lbl in step_dict[s]]

        self.masking_value = masking_value
        self.masking = masking

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}

        if train:
            self.inverted_order[255] = masking_value
        else:
            self.set_up_void_test()

        if masking:
            tmp_labels = self.labels + [255]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order

        # if not (train and self.weakly):
        mapping = np.zeros((256,))
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]

        self.transform_lbl = LabelTransform(mapping)
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)
    
    def get_mapping_dict(self):
        return self.inverted_order

    def set_up_void_test(self):
        self.inverted_order[255] = 255

    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        if index < len(self):
            data = self.dataset[index]
            img, lbl, lbl_1h = data[0], data[1], data[2]
            img, lbl = self.transform(img, lbl)
            lbl = self.transform_lbl(lbl)
            l1h = self.transform_1h(lbl_1h)

            return img, lbl, l1h
        else:
            raise ValueError("absolute value of index should not exceed dataset length")

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, root, train, indices, saliency=False, pseudo=None):
        raise NotImplementedError


class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return from_numpy(self.mapping[x])


class LabelSelection:
    def __init__(self, order, labels, masking):
        order = np.array(order)
        order = order[order != 0]
        order -= 1  # scale to match one-hot index.
        self.order = order
        if masking:
            self.masker = np.zeros((len(order)))
            self.masker[-len(labels)+1:] = 1
        else:
            self.masker = np.ones((len(order)))

    def __call__(self, x):
        x = x[self.order] * self.masker
        return x

class ImageNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 imagenet_dict,
                 step_dict,
                 mapping_dict,
                 step,
                 external_size,
                 external_mode,
                 labels_cum,
                 transform=None,
                 split_root=None):

        self.transform = transform
        self.tot_classes = len(labels_cum) - 1
        self.nb_old_classes = 0
        
        full_imagenet_dset = datasets.ImageFolder(root=os.path.join(root, 'train'))
        labels_old = [lbl for s in range(step) for lbl in step_dict[s]][1:] # exclude the background class
        self.nb_old_classes = len(labels_old)

        if external_mode == 'old':
            labels = labels_old
        elif external_mode == 'new':
            labels = list(step_dict[step])
        elif external_mode == 'both':
            labels = list(step_dict[step]) + labels_old

        wnids = [imagenet_dict[label] for label in labels]
        wnids_to_idxs = [[full_imagenet_dset.class_to_idx[w] for w in wnid] for wnid in wnids]
        
        # check if the previous-idxs are already saved in the memory or not
        mem_fname = os.path.join(split_root, f'train-ext-mem-{step}.npy')
        wnids_item_idxes = list()
        # if exists then load from the saved npy files
        if os.path.exists(mem_fname):
            wnids_item_idxes = np.load(mem_fname)
        else:
            # imgnet_cls_ids
            wnids_item_idxes = [random.sample(list(np.where(np.any([np.asarray(full_imagenet_dset.targets) == item for item in imgnet_cls_ids], axis=0))[0]), external_size) for imgnet_cls_ids in wnids_to_idxs]
            wnids_item_idxes = np.concatenate(wnids_item_idxes, axis=0)
            if distributed.get_rank() == 0:
                np.save(mem_fname, np.array(wnids_item_idxes, dtype=int))

        self.dataset = torch.utils.data.Subset(full_imagenet_dset, wnids_item_idxes)

        inverted_imagenet_dict = dict()
        for k, values in imagenet_dict.items():
            for v in values:
                inverted_imagenet_dict[v] = k

        self.target_transform = dict()
        for k, v in inverted_imagenet_dict.items():
            if v in mapping_dict:
                self.target_transform[full_imagenet_dset.class_to_idx[k]] = mapping_dict[v]

    def __getitem__(self, index):
        img = self.dataset[index][0]
        target = self.dataset[index][1]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform[target]
            # convert to one-hot label
            target = F.one_hot(torch.tensor(target-1), num_classes=self.tot_classes)
            target = self.mask_old_classes(target)
            return img, target
        return img

    def __len__(self):
        return len(self.dataset)
    
    def mask_old_classes(self, l1h):
        masker = torch.ones(l1h.shape)
        masker[:self.nb_old_classes] = 0
        return masker * l1h

class FineGrainedIncrementalSegmentation(data.Dataset):
    def __init__(
            self, 
            root, 
            step_dict, 
            train=True, 
            transform=None, 
            masking=True, 
            masking_value=0, 
            step=0
    ):

        self.dataset = self.make_dataset(root=root, train=train)
        self.transform = transform
        self.train = train
        self.classes = self.class_dict()
        self.imagenet_classes = self.imagenet_wnid_dict()

        self.step_dict = step_dict
        self.labels = []
        self.labels_old = []
        self.step = step

        # it enumerates all the classes sequentially for all the tasks observed so far
        self.order = [cl for s in sorted(step_dict) for cl in step_dict[s]]

        if step > 0:
            self.labels = [self.order[0]] + list(step_dict[step])
        else:
            self.labels = list(step_dict[step])
        
        self.labels_old = [cl for s in range(step) for cl in step_dict[s]]
        
        self.masking_value = masking_value
        self.masking = masking

        if self.train:
            idxs = filter_images(self.dataset, self.labels, self.labels_old)
        else:
            idxs = None

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}

        if train:
            self.inverted_order[255] = masking_value
        else:
            self.inverted_order[255] = 255
        
        if masking:
            temp_labels = self.labels + [255]
            mapping_dict = {x: self.inverted_order[x] for x in temp_labels}
        else:
            mapping_dict = self.inverted_order
        
        mapping = np.zeros((256,))
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]
        
        self.transform_lbl = LabelTransform(mapping)
        self.transform_1h = LabelSelection(self.order, self.labels, self.masking)

        # make the subset of the dataset
        if idxs is not None:
            self.dataset = torch.utils.data.Subset(self.dataset, idxs)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        img, target, l1h = data[0], data[1], data[2]

        if self.transform is not None:
            img, target = self.transform(img, target)

        # target transforms
        target = self.transform_lbl(target)
        l1h = self.transform_1h(l1h)

        return img, target, l1h
    
    def __len__(self):
        return len(self.dataset)
    
    def get_mapping_dict(self):
        return self.inverted_order
        
    def make_dataset(self, root, train):
        raise NotImplementedError


class FSSDataset(data.Dataset):
    def __init__(self,
                 root,
                 task,
                 train=True,
                 transform=None,
                 masking=False):

        self.full_data = self.make_dataset(root, train)
        self.transform = transform

        step = self.step = task.step
        self.order = task.order
        self.labels = task.get_novel_labels()
        self.labels_old = task.get_old_labels(bkg=False)
        self.labels_fut = task.get_future_labels()

        assert not any(l in self.labels_old for l in self.labels), "labels and labels_old must be disjoint sets"
        assert not any(l in self.labels_fut for l in self.labels), "labels and labels_fut must be disjoint sets"

        self.masking_value = masking_value = 255
        self.class_to_images = {}

        self.inverted_order = {lb: self.order.index(lb) for lb in self.order}
        if train:
            self.inverted_order[255] = 255
            self.inverted_order[0] = 0
        else:
            self.inverted_order[0] = 0
            self.set_up_void_test()
        
        self.multi_idxs = False # to be checked
        if not train:
            # in test we always use all images
            idxs = list(range(len(self.full_data)))
            # we mask unseen classes to 255. Usually masking is False and value=255.
            target_transform = self.get_mapping_transform(self.labels, masking=masking, masking_value=masking_value)
            self.class_to_images = self.full_data.class_to_images # to be checked
        
        elif step == 0 or task.nshot == -1:
            # we filter images containing pixels of unseen classes.
            idxs = {x for x in range(len(self.full_data))}
            if task.disjoint:
                for cl, img_set in self.full_data.class_to_images.items():
                    if cl not in self.labels and (cl != 0):
                        idxs = idxs.difference(img_set)
            idxs = list(idxs)
            # this is useful to reorder the labels (not to mask since we already excluded the to-mask classes)
            target_transform = self.get_mapping_transform(self.labels, masking, masking_value)
            # this is helpful in case we need to sample images of some class
            index_map = {idx: new_idx for new_idx, idx in enumerate(idxs)}
            for cl in self.labels:
                self.class_to_images[cl] = []
                for idx in self.full_data.class_to_images[cl]:
                    if idx in index_map:
                        self.class_to_images[cl].append(index_map[idx])
        else: # few shot learning
            self.multi_idxs = True
            idxs = {}
            target_transform = {}
            ishot = task.ishot
            nshot = task.nshot
            if task.input_mix == 'both':
                idxs[0] = []
                for cl in self.labels_old:
                    # 20 is max of nshot - taken from SPNet code
                    idxs[0].extend(self.full_data.class_to_images[cl][ishot*20: ishot*20+nshot])
                target_transform[0] = self.get_mapping_transform(self.labels_old, masking=True,
                                                                 masking_value=masking_value)
            
            for i, cl in enumerate(self.labels):
                images_of_class = self.full_data.class_to_images[cl]
                if FILTER_FSL:
                    # filter images containing unseen or actual classes
                    for cl_, img_set in self.full_data.class_to_images.items():
                        if cl_ != cl and cl_ not in self.labels_old and (cl_ != 0):
                            images_of_class = images_of_class.difference(img_set)
                idxs[i+1] = images_of_class[ishot*20: ishot*20+nshot]
                lbls = [cl] if masking else self.labels_old + [cl]
                # this is useful to reorder the labels (not to mask since we already excluded the to-mask classes)
                target_transform[i+1] = self.get_mapping_transform(lbls, masking=True, masking_value=masking_value)

        # make a subset of the dataset
        self.indices = []
        if not self.multi_idxs: # step=0 or test
            self.dataset = Subset(self.full_data, idxs, transform=transform, target_transform=target_transform)
        else:
            dts_list = []
            count = 0
            if 0 in idxs:
                dts_list.append(Subset(self.full_data, idxs[0],
                                       transform=transform, target_transform=target_transform[0]))
                for cl in self.labels_old:
                    self.class_to_images[cl] = []
                    for new_idx, idx in enumerate(idxs):
                        if idx in self.full_data.class_to_images[cl]:
                            self.class_to_images[cl].append(new_idx)
                count += len(idxs[0])
            for i in range(1, len(self.labels)+1):
                dts_list.append(Subset(self.full_data, idxs[i],
                                       transform=transform, target_transform=target_transform[i]))
                self.class_to_images[self.labels[i-1]] = list(range(count, count+len(idxs[i])))
                count += len(idxs[i])

            self.dataset = ConcatDataset(dts_list)

        # LabelSelection expects bkg class in the self.labels list
        self.labels.insert(0, 0)
        self.transform_1h = LabelSelection(self.order, self.labels, masking=True)
    
    def set_up_void_test(self):
        self.inverted_order[255] = 255
    
    def get_mapping_transform(self, labels, masking, masking_value):
        # set up the mapping
        # if masking=True, old classes become masking_value, except the bkg.
        # if masking=False, all seen classes are reordered according to the order. No seen class is excluded.
        if masking:
            tmp_labels = labels + [255, 0]
            mapping_dict = {x: self.inverted_order[x] for x in tmp_labels}
        else:
            mapping_dict = self.inverted_order
        mapping = np.full((256,), masking_value, dtype=np.uint8)
        for k in mapping_dict.keys():
            mapping[k] = mapping_dict[k]
        target_transform = LabelTransform(mapping)

        return target_transform
    
    def __getitem__(self, index):
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        
        img, lbl, lbl_1h = self.dataset[index]
        l1h = self.transform_1h(lbl_1h)

        return img, lbl, l1h
    
    def __len__(self):
        return len(self.dataset)
    
    def make_dataset(self, root, train):
        raise NotImplementedError
    
    def get_k_image_of_class(self, cl, k):
        assert cl < len(self.order) and cl != 0, f"Class must be in the actual task! Obtained {cl}"

        cl = self.order[cl]  # map to original mapping!
        assert len(self.class_to_images[cl]) >= k, f"There are no K images available for class {cl}."
        id_list = random.sample(self.class_to_images[cl], k=k)
        ret_images = []
        for i in id_list:
            ret_images.append(self[i])
        return ret_images

        
        
        
        