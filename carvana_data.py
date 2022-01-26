# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 22:48:10 2021

@author: alida
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

#image_dir=r'C:\Users\alida\.spyder-py3\image_data\Carvana\train'
#mask_dir=r'C:\Users\alida\.spyder-py3\image_data\Carvana\train_masks'

class Carvanadata(Dataset):
    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir=image_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.images=os.listdir(image_dir)
        self.masks=os.listdir(mask_dir)
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        img_path=os.path.join(self.image_dir,self.images[idx])
        mask_path=os.path.join(self.mask_dir,self.images[idx][:-1]) #.replace(".jpg","_mask.gif"))
       # print(mask_path)
        image=np.array(Image.open(img_path).convert("L"))
        mask=np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #mask[mask==255]=1.0
        if self.transform is not None:
          augumentation=self.transform(image=image, mask=mask)
          image=augumentation["image"]
          mask=augumentation["mask"] 
        return image, mask
    
    


