import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import cv2

class NPCDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = read_image(img_path)
        # image = cv2.imread(img_path)
        # image = np.moveaxis(image,-1,0)
        # image = torch.tensor(image)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))

        #if self.transform:
        #    image = self.transform(image)
        
        # if self.target_transform:
        #     label = self.target_transform(label)
        
        return image, label