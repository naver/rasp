tasks = {}

tasks['coco'] = {
    "offline":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
                88, 89, 90],
        },
    "voc":
        {
            0: [0, 8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78,
                79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],  # 17.443 train img
            1: [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]  # voc classes
        },
    "20-0": 
        {
            0: [2, 3, 4, 6, 7, 8, 10, 11, 13, 15, 16, 17, 19, 20, 21, 23, 24,
                25, 28, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44, 47, 48, 49, 51,
                52, 53, 55, 56, 57, 59, 60, 61, 63, 64, 65, 70, 72, 73, 75, 76, 77,
                79, 80, 81, 84, 85, 86, 88, 89, 90],
            1: [1, 5, 9, 14, 18, 22, 27, 33, 37, 41, 46, 50, 54, 58, 62, 67, 74, 78, 82, 87]
        },
    "20-1":
        {
            0: [1, 3, 4, 5, 7, 8, 9, 11, 13, 14, 16, 17, 18, 20, 21, 22, 24,
                25, 27, 31, 32, 33, 35, 36, 37, 39, 40, 41, 43, 44, 46, 48, 49, 50,
                52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 65, 67, 72, 73, 74, 76, 77,
                78, 80, 81, 82, 85, 86, 87, 89, 90],
            1: [2, 6, 10, 15, 19, 23, 28, 34, 38, 42, 47, 51, 55, 59, 63, 70, 75, 79, 84, 88]
        },
    "20-2":
        {
            0: [1, 2, 4, 5, 6, 8, 9, 10, 13, 14, 15, 17, 18, 19, 21, 22, 23,
                25, 27, 28, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 46, 47, 49, 50,
                51, 53, 54, 55, 57, 58, 59, 61, 62, 63, 65, 67, 70, 73, 74, 75, 77,
                78, 79, 81, 82, 84, 86, 87, 88, 90],
            1: [3, 7, 11, 16, 20, 24, 31, 35, 39, 43, 48, 52, 56, 60, 64, 72, 76, 80, 85, 89]
        },
    "20-3":
        {
            0: [1, 2, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16, 18, 19, 20, 22, 23,
                24, 27, 28, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 48, 50,
                51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 67, 70, 72, 74, 75, 76,
                78, 79, 80, 82, 84, 85, 87, 88, 89],
            1: [4, 8, 13, 17, 21, 25, 32, 36, 40, 44, 49, 53, 57, 61, 65, 73, 77, 81, 86, 90]
        },
}

tasks['voc'] = {
    "offline":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    "19-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            1: [20],
        },
    "19-1b":
        {
            0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [5],
        },
    "15-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16, 17, 18, 19, 20]
        },
    "15-5b":
        {
            0: [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20],
            1: [3, 4, 10, 14, 18] # bird, boat, cow, motorbike, sofa
        },
    "15-5c":
        {
            0: [0, 1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20],
            1: [2, 6, 9, 12, 15] # bicycle, bus, chair, dog, person
        },
    "15-5d": # the new classes are the ones where groupvit has the topmost scores
        {
            0: [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 14, 15, 16, 18, 19, 20],
            1: [6, 8, 10, 12, 17] # bus, cat, cow, dog, sheep
        },
    "15-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16],
            2: [17],
            3: [18],
            4: [19],
            5: [20]
        },
    "18-2":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            1: [19, 20], # train, tvmonitor
        },
    "18-2b":
        {
            0: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [1, 11], # aeroplane, dinningtable
        },
    "18-2c":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20],
            1: [7, 13], # car, horse
        },
    "18-2d":
        {
            0: [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [2, 10], # bicycle, cow
        },
    "10-5-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15],
            2: [16, 17, 18, 19, 20]
        },
    "10-2":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12],
            2: [13, 14],
            3: [15, 16],
            4: [17, 18],
            5: [19, 20]
        },
    "10-10":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        },
    "10-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11],
            2: [12],
            3: [13],
            4: [14],
            5: [15],
            6: [16],
            7: [17],
            8: [18],
            9: [19],
            10: [20]
        },
    # few shot segmentation tasks
    "5-0":
        {
            0: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [1, 2, 3, 4, 5]
        },
    "5-1":
        {
            0: [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [6, 7, 8, 9, 10]
        },
    "5-2":
        {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
            1: [11, 12, 13, 14, 15]
        },
    "5-3":
        {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16, 17, 18, 19, 20]
        }
}

tasks['coco-voc'] = {
    "voc":
        {
            0: [0, 8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78,
                79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],  # coco classes
            1: [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]  # voc classes
        },
    "voc-5":
        {
            0: [0, 8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78,
                79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],  # coco classes
            1: [1, 2, 3, 4, 5],
            2: [6, 7, 9, 16, 17],
            3: [18, 19, 20, 21, 44],
            4: [62, 63, 64, 67, 72]
        },
    "voc-2":
        {
            0: [0, 8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78,
                79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],  # coco classes
            1: [1, 2],
            2: [3, 4],
            3: [5, 6],
            4: [7, 9],
            5: [16, 17],
            6: [18, 19],
            7: [20, 21],
            8: [44, 62],
            9: [63, 64],
            10: [67, 72]
        }
        # ignore_labels = [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]  # starting from 1=person
}

tasks['cub'] = {
    'offline':
        {
            0: [x for x in range(201)],
        },
    '50-50':
        {
            0: [x for x in range(51)],
            1: [x for x in range(51, 101)],
            2: [x for x in range(101, 151)],
            3: [x for x in range(151, 201)],
        },
    '50-10':
        {
            0: [0, 134,  83,   5, 133,  58, 158,  92, 130, 171,  75, 175,  85,  81,
                87, 124, 150, 189, 195, 121,  33,  18,  82,  50,  46,  28, 147,
                7, 143, 104,  79, 111,  78,  36, 198,  77,  57, 145, 173, 163,
                43, 196, 119,  29, 139, 176, 151,   2,  25,  45,  16],
            1: [154, 107, 178, 114,  20,  19, 108, 146, 184, 194],
            2: [159,  32, 123,  94,  24, 137, 193,  71,  68,  39],
            3: [27,  40,   9,  12,  55, 160, 110, 165, 177, 181],
            4: [199,  64,  30,  76,  67,  15,  65,  80,  48,  23],
            5: [72, 103,  99, 132,  49, 140, 109,  95,  56, 164],
            6: [51,  91,  26, 182, 191, 190,  17, 126,  10,  73],
            7: [170,   8,  13, 183, 185, 125, 192,  38, 162, 172],
            8: [37, 144, 186, 136, 106, 169,  34, 117,  70, 131],
            9: [179, 155,  54, 102, 188, 168,  74, 167, 135, 100],
            10: [47, 180,  41, 161, 200,  86,  14,  59,  88,  93],
            11: [120, 112, 127, 197, 122, 142, 128,  61, 129, 148],
            12: [116,  98,  62, 115,  60, 113, 149,   1,   6,  35],
            13: [42, 152,  53, 157,  22,  44,  69,  90,  52,  84],
            14: [153,  11, 105, 138,  21, 101,  97,  89,   3,  96],
            15: [187, 156, 166, 118,   4,  31, 141,  63,  66, 174]
        }
        
}

def get_transformed_task_labels(labels):
    transformation_dict = dict(zip(labels, range(len(labels))))
    return transformation_dict


def get_task_list():
    return [task for ds in tasks.keys() for task in tasks[ds].keys()]


def get_task_labels(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old, f'{dataset}/{name}'


def get_task_dict(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    class_map = {s: task_dict[s] for s in range(step+1)}
    return class_map

def get_order(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError('The task or the dataset is not supported')
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    order = [cl for s in range(step + 1) for cl in task_dict[s]]
    return order


def get_per_task_classes(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step+1)]
    return classes

# for FSS
class Task:
    def __init__(self, opts):
        self.step = opts.step
        self.dataset = opts.dataset
        self.task = opts.task
        if self.task not in tasks[self.dataset]:
            raise NotImplementedError(f'The task {self.task} is not present in {self.dataset}')
        self.task_dict = tasks[self.dataset][self.task]
        assert self.step in self.task_dict.keys(), f'You should provide a valid step! [{self.step} is out of range]'
        self.order = [cl for s in range(self.step + 1) for cl in self.task_dict[s]]

        self.disjoint = True

        self.nshot = opts.nshot if self.step > 0 else -1
        self.ishot = opts.ishot
        self.input_mix = opts.input_mix # novel/both

        self.num_classes = len(self.order)

        # add the background
        self.order.insert(0, 0)
        self.num_classes += 1
    
    def get_order(self):
        return self.order
    
    def get_future_labels(self):
        return [cl for s in self.task_dict.keys() for cl in self.task_dict[s] if s > self.step]
    
    def get_novel_labels(self):
        return list(self.task_dict[self.step])
    
    def get_old_labels(self, bkg=True):
        if bkg:
            return [0] + [cl for s in range(self.step) for cl in self.task_dict[s]]
        else:
            return [cl for s in range(self.step) for cl in self.task_dict[s]]
    
    def get_task_dict(self):
        return {s: self.task_dict[s] for s in range(self.step + 1)}
    
    # return a list containing number of classes in each task
    def get_n_classes(self):
        r = [len(self.task_dict[s]) for s in range(self.step + 1)]
        # consider background
        r[0] += 1
        return r