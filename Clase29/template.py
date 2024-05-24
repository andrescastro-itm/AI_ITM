template = '''import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):

    def __init__(self, target_transform=None):

        self.path = 'dir_path'
        self.transform = transforms
        self.target_transform = target_transform
        self.dims = (dims)
        self.ext = []

        subdirs = next(os.walk(self.path))[1]

        L = []
        for i, class_ in enumerate(subdirs):
            imgs = next(os.walk(os.path.join(self.path, class_)))[2]

            for img in imgs:
                if img[-3:] in self.ext:
                    pathimg = os.path.join(self.path, class_, img)
                    L.append([pathimg, i])

        self.df = pd.DataFrame(L, columns=['Path', 'Class'])

    def __len__(self):

        return self.df.shape[0]

    def __getitem__(self, idx):

        img_path = self.df.at[idx, 'Path']
        label = self.df.at[idx, 'Class']

        image = Image.open(img_path)
        image = image.resize(self.dims)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
'''
