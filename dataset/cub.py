import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import pickle5 as pkl
from torch import distributed
from .dataset import FineGrainedIncrementalSegmentation
from .utils import get_bert_similarity

CUB_CLASSES = {
    0: 'background', 1: 'Black footed Albatross', 2: 'Laysan Albatross', 3: 'Sooty Albatross', 4: 'Groove billed Ani', 5: 'Crested Auklet', 
    6: 'Least Auklet', 7: 'Parakeet Auklet', 8: 'Rhinoceros Auklet', 9: 'Brewer Blackbird', 10: 'Red winged Blackbird', 11: 'Rusty Blackbird', 
    12: 'Yellow headed Blackbird', 13: 'Bobolink', 14: 'Indigo Bunting', 15: 'Lazuli Bunting', 16: 'Painted Bunting', 17: 'Cardinal', 
    18: 'Spotted Catbird', 19: 'Gray Catbird', 20: 'Yellow breasted Chat', 21: 'Eastern Towhee', 22: 'Chuck will Widow', 23: 'Brandt Cormorant', 
    24: 'Red faced Cormorant', 25: 'Pelagic Cormorant', 26: 'Bronzed Cowbird', 27: 'Shiny Cowbird', 28: 'Brown Creeper', 29: 'American Crow', 
    30: 'Fish Crow', 31: 'Black billed Cuckoo', 32: 'Mangrove Cuckoo', 33: 'Yellow billed Cuckoo', 34: 'Gray crowned Rosy Finch', 
    35: 'Purple Finch', 36: 'Northern Flicker', 37: 'Acadian Flycatcher', 38: 'Great Crested Flycatcher', 39: 'Least Flycatcher', 
    40: 'Olive sided Flycatcher', 41: 'Scissor tailed Flycatcher', 42: 'Vermilion Flycatcher', 43: 'Yellow bellied Flycatcher', 
    44: 'Frigatebird', 45: 'Northern Fulmar', 46: 'Gadwall', 47: 'American Goldfinch', 48: 'European Goldfinch', 49: 'Boat tailed Grackle', 
    50: 'Eared Grebe', 51: 'Horned Grebe', 52: 'Pied billed Grebe', 53: 'Western Grebe', 54: 'Blue Grosbeak', 55: 'Evening Grosbeak', 
    56: 'Pine Grosbeak', 57: 'Rose breasted Grosbeak', 58: 'Pigeon Guillemot', 59: 'California Gull', 60: 'Glaucous winged Gull', 
    61: 'Heermann Gull', 62: 'Herring Gull', 63: 'Ivory Gull', 64: 'Ring billed Gull', 65: 'Slaty backed Gull', 66: 'Western Gull', 
    67: 'Anna Hummingbird', 68: 'Ruby throated Hummingbird', 69: 'Rufous Hummingbird', 70: 'Green Violetear', 71: 'Long tailed Jaeger', 
    72: 'Pomarine Jaeger', 73: 'Blue Jay', 74: 'Florida Jay', 75: 'Green Jay', 76: 'Dark eyed Junco', 77: 'Tropical Kingbird', 
    78: 'Gray Kingbird', 79: 'Belted Kingfisher', 80: 'Green Kingfisher', 81: 'Pied Kingfisher', 82: 'Ringed Kingfisher', 
    83: 'White breasted Kingfisher', 84: 'Red legged Kittiwake', 85: 'Horned Lark', 86: 'Pacific Loon', 87: 'Mallard', 
    88: 'Western Meadowlark', 89: 'Hooded Merganser', 90: 'Red breasted Merganser', 91: 'Mockingbird', 92: 'Nighthawk', 
    93: 'Clark Nutcracker', 94: 'White breasted Nuthatch', 95: 'Baltimore Oriole', 96: 'Hooded Oriole', 97: 'Orchard Oriole', 
    98: 'Scott Oriole', 99: 'Ovenbird', 100: 'Brown Pelican', 101: 'White Pelican', 102: 'Western Wood Pewee', 103: 'Sayornis', 
    104: 'American Pipit', 105: 'Whip poor Will', 106: 'Horned Puffin', 107: 'Common Raven', 108: 'White necked Raven', 
    109: 'American Redstart', 110: 'Geococcyx', 111: 'Loggerhead Shrike', 112: 'Great Grey Shrike', 113: 'Baird Sparrow', 
    114: 'Black throated Sparrow', 115: 'Brewer Sparrow', 116: 'Chipping Sparrow', 117: 'Clay colored Sparrow', 118: 'House Sparrow', 
    119: 'Field Sparrow', 120: 'Fox Sparrow', 121: 'Grasshopper Sparrow', 122: 'Harris Sparrow', 123: 'Henslow Sparrow', 
    124: 'Le Conte Sparrow', 125: 'Lincoln Sparrow', 126: 'Nelson Sharp tailed Sparrow', 127: 'Savannah Sparrow', 128: 'Seaside Sparrow', 
    129: 'Song Sparrow', 130: 'Tree Sparrow', 131: 'Vesper Sparrow', 132: 'White crowned Sparrow', 133: 'White throated Sparrow', 
    134: 'Cape Glossy Starling', 135: 'Bank Swallow', 136: 'Barn Swallow', 137: 'Cliff Swallow', 138: 'Tree Swallow', 139: 'Scarlet Tanager', 
    140: 'Summer Tanager', 141: 'Artic Tern', 142: 'Black Tern', 143: 'Caspian Tern', 144: 'Common Tern', 145: 'Elegant Tern', 
    146: 'Forsters Tern', 147: 'Least Tern', 148: 'Green tailed Towhee', 149: 'Brown Thrasher', 150: 'Sage Thrasher', 
    151: 'Black capped Vireo', 152: 'Blue headed Vireo', 153: 'Philadelphia Vireo', 154: 'Red eyed Vireo', 155: 'Warbling Vireo', 
    156: 'White eyed Vireo', 157: 'Yellow throated Vireo', 158: 'Bay breasted Warbler', 159: 'Black and white Warbler', 
    160: 'Black throated Blue Warbler', 161: 'Blue winged Warbler', 162: 'Canada Warbler', 163: 'Cape May Warbler', 
    164: 'Cerulean Warbler', 165: 'Chestnut sided Warbler', 166: 'Golden winged Warbler', 167: 'Hooded Warbler', 
    168: 'Kentucky Warbler', 169: 'Magnolia Warbler', 170: 'Mourning Warbler', 171: 'Myrtle Warbler', 172: 'Nashville Warbler', 
    173: 'Orange crowned Warbler', 174: 'Palm Warbler', 175: 'Pine Warbler', 176: 'Prairie Warbler', 177: 'Prothonotary Warbler', 
    178: 'Swainson Warbler', 179: 'Tennessee Warbler', 180: 'Wilson Warbler', 181: 'Worm eating Warbler', 182: 'Yellow Warbler', 
    183: 'Northern Waterthrush', 184: 'Louisiana Waterthrush', 185: 'Bohemian Waxwing', 186: 'Cedar Waxwing', 
    187: 'American Three toed Woodpecker', 188: 'Pileated Woodpecker', 189: 'Red bellied Woodpecker', 
    190: 'Red cockaded Woodpecker', 191: 'Red headed Woodpecker', 192: 'Downy Woodpecker', 193: 'Bewick Wren', 
    194: 'Cactus Wren', 195: 'Carolina Wren', 196: 'House Wren', 197: 'Marsh Wren', 198: 'Rock Wren', 
    199: 'Winter Wren', 200: 'Common Yellowthroat'
}

class CUB200Segmentation(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        if train:
            split = 'train'
        else:
            split = 'val'
        
        self.root = os.path.expanduser(root)
        self.transform = transform

        cub_root = os.path.join(self.root, 'cub/')
        split_dir = os.path.join(cub_root, 'split/')

        if not os.path.isdir(cub_root):
            raise RuntimeError(f'Dataset not found in {cub_root}')
        
        split_f = os.path.join(split_dir, f'{split}.txt')
        if not os.path.exists(split_f):
            raise RuntimeError(f'The file {split_f} containing the image set is not found!')
        
        # remove leading \n
        with open(split_f, 'r') as f:
            file_names = [x[:-1] for x in f.readlines()]
        
        images_dir = 'images'
        ann_dir = 'annotations'

        self.images = [(os.path.join(cub_root, images_dir, i + '.jpg'),
                        os.path.join(cub_root, ann_dir, i + '.png')) for i in file_names]
        
        if train:
            self.img_lvl_labels = np.load(os.path.join(split_dir, 'cub_fss_1h_labels_train.npy'))
        else:
            self.img_lvl_labels = np.load(os.path.join(split_dir, 'cub_fss_1h_labels_val.npy'))
        
        assert len(self.img_lvl_labels) == len(self.images), "Number of images and one hot labels annotations do not match. Something is wrong!"

    
    def __getitem__(self, index):
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        img_lvl_label = self.img_lvl_labels[index]

        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target, img_lvl_label
    
    def __len__(self):
        return len(self.images)

class CUB200SegmentationIncremental(FineGrainedIncrementalSegmentation):
    def make_dataset(self, root, train):
        full_cub = CUB200Segmentation(root, train)
        return full_cub
    
    def class_dict(self):
        """
        Returns the class names in the CUB data set
        """
        return CUB_CLASSES
    
    def class_affinity(self, scaling):
        """
        Returns the affinity between a given pair of classes
        """
        # for bert based similarity
        return get_bert_similarity(self.class_dict(), 384, scaling=scaling)
    
    def get_order(self):
        # Returns in order all the classes seen so far
        return self.order
    
    def imagenet_wnid_dict(self):
        """
        Returns the imagenet and CUB corresponding class dictionary
        """
        return None
    