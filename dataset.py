import glob
import random
import os
from torch._C import Value
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, root, transforms_ = None, unaligned = False, mode = 'train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        #img A,B
        self.files_A = sorted(glob.glob(os.path.join(root, mode+'A')+ '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, mode+'B')+ '/*.*'))

        #label A,B
        self.label_A = [line.rstrip() for line in open('datasets/data/label_A.txt','r', encoding='utf-8')]
        self.label_B = [line.rstrip() for line in open('datasets/data/label_B.txt','r', encoding='utf-8')]

        self.train_dataA = []
        self.train_dataB = []

        random.shuffle(self.label_A)
        random.shuffle(self.label_B)
        
        #label A
        for i, line in enumerate(self.label_A):
            split = line.split()
            filename = split[0]
            values = split[1:]
            values = list(map(float, values))
            values = np.array(values)
            label = torch.from_numpy(values)
            self.train_dataA.append([filename, label])
        #label B
        for i, line in enumerate(self.label_B):
            split = line.split()
            filename = split[0]
            values = split[1:]
            values = list(map(float, values))
            values = np.array(values)
            label = torch.from_numpy(values)

            self.train_dataB.append([filename, label])

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        A_label = self.train_dataA[index % len(self.files_A)][1]
    
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
        B_label = self.train_dataB[index % len(self.files_B)][1]
        return {'A': item_A, 'B':item_B, 'A_label':A_label, 'B_label':B_label}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))