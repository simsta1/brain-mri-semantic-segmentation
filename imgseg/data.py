import torch
from torch.utils.data import DataLoader, Dataset
import PIL
import os
from torchvision import transforms as transforms
import pandas as pd
import numpy as np
import pickle


class SegmentationDataset(Dataset):
    """
    Class defines custom Dataset as a workaround to the ImageFolder class
    """

    def __init__(self, image_dir: str, mask_dir: str, df: pd.DataFrame, 
                 transform: 'transforms.Compose' = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.df = df
        self.transforms = transform
        
    def __len__(self):
        """Returns length of Dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """Returns one item from dataset"""
        image, mask = self.df.loc[idx, ['image', 'mask']].values
        image = PIL.Image.open(os.path.join(self.image_dir, image))
        mask = PIL.Image.open(os.path.join(self.mask_dir, mask))
        
        tensor_transform = transforms.ToTensor()
        
        if self.transforms:
            image = self.transforms(image)
        else:
            image = tensor_transform(image)
        mask = tensor_transform(mask)
        
        return image, mask
        
        
def get_dataloader(image_dir: str, mask_dir: str, df: pd.DataFrame,  
                   batch_size: int, workers: int, 
                   transformations: 'transforms.Compose' = None,
                   **dlkwargs):

    custom_dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, df=df, 
                                        transform=transformations)
    return DataLoader(dataset=custom_dataset,
                      batch_size=batch_size,
                      num_workers=workers, 
                      **dlkwargs)