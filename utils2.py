# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:30:48 2022

@author: sem
"""

import torch
import cv2
import torchvision
from carvana_data import Carvanadata
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torchvision.transforms
def save(status,path):
  torch.save(status, path)
  print("saving the model and optimizer parameters ")

def save_best(best_status,best_path):
  torch.save(best_status, best_path)
  print("saving the BEST model ")




def load(path,model,optimizer):
  status=torch.load(path)
  model.load_state_dict(status["state_dict"])
  optimizer.load_state_dict(status["optimizer"])
  best_score=status["score"]
  print(f'loading all trained parameters with dice of {best_score}')
  return best_score
  
def get_loaders(
    image_dir,
    mask_dir,
    val_image_dir,
    val_mask_dir,
    batch,
    train_transform,
    val_transform,
#    num_workers, #=2,
#    pin_memory, #=True,
):
    train_ds = Carvanadata(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
#        num_workers=num_workers,
#        pin_memory=pin_memory,
    )

    val_ds = Carvanadata(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

 #   val_un_ds=torch.tensor(val_array_ds)
  #  valid_normal=torchvision.transforms.Compose([
   #      torchvision.transforms.Normalize(179,54)                                        

    #   ])
    #val_ds=valid_normal(val_un_ds).float().unsqueeze(1)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
#        num_workers=num_workers,
#        pin_memory=pin_memory,
    )

    return train_loader  , val_loader

def check_acc(loader,model,device="cuda"):
   
   valid_normal=torchvision.transforms.Compose([
         
         torchvision.transforms.Normalize(179,54)                                        

       ])
   num_correct=0
   num_pixels=0
   dice=0
   model.eval()
   with torch.no_grad():
     for x,y in loader:
       x=torch.tensor(x,dtype=torch.float32).to(device)
       x=valid_normal(x).unsqueeze(1)
       #print(f'shape of input tensor is {x.shape}')
       with torch.no_grad():
        predict=model(x)
        sf=nn.Softmax(dim=1)
        soft_preds=sf(predict)
        preds=torch.argmax(soft_preds,dim=1,keepdim=False)
       y=y.unsqueeze(1).to(device)
       # print(y.shape)
      ##### preds=torch.sigmoid(model(x))
      ####### preds=(preds>0.5).float().to("cpu")
      
       preds=torch.flatten(preds).to("cpu")
       y=torch.flatten(y).to("cpu")

       preds=np.array(preds)
       y=np.array(y)
       
       num_correct+=(preds==y).sum()
       #print(num_correct)
             #######num_pixels+=torch.numel(preds)
       num_pixels+=preds.size
       #print(num_pixels)
       ######dice+=(2*((preds*y).sum()))/((preds + y).sum())


   accp=(num_correct/num_pixels)*100
   print(f'we found {num_correct} correct in {num_pixels} total with validation accuracy of %{accp:.2f}')
   #print(f'Dice score is {dice/len(loader):.2f}')
   
   model.train()
   #return dice

def save_pred(loader, model, folder="/content/gdrive/MyDrive/DL_files/Unet_multi/save_preds/"
 ,device="cuda"):
  
 # val_un_ds=torch.tensor(val_array_ds)
  valid_normal=torchvision.transforms.Compose([
         
         torchvision.transforms.Normalize(179,54)                                        

       ])
    
    
  model.eval()
  for idx, (x,y) in enumerate(loader):
       x=torch.tensor(x,dtype=torch.float32).to(device)
       x_copy=x
       xprint=x_copy
       
       print(f'x print dimension is {xprint.shape}')
       x=valid_normal(x).unsqueeze(1)
       
       #x=x.unsqueeze(1)
       #print(f'idx is {idx}')
       #print(f'loader length is {len(loader)}')
       with torch.no_grad():
        predict=model(x)
        
        sf=nn.Softmax(dim=1)
        soft_preds=sf(predict)
        
        
        preds=torch.argmax(soft_preds,dim=1,keepdim=False)
       # print(preds.dtype)
       # print(preds.shape)
       # print(torch.max(preds))
       # preds.type(torch.LongTensor)
        #preds=torch.sigmoid(model(x))
        #preds=(preds>0.5).float()
        channel=preds[:,0,0]
        ch=torch.numel(channel)
       # oneimage=preds[1,:,:]
        print (f'channel number is {ch}')
        
       for cc in range(0, ch):
         image=preds[cc,:,:].to("cpu")
         nimage=image.numpy()
         cv2.imwrite(f"{folder}pred_{idx}{cc}.tiff",nimage)
         
         gtruth=y[cc,:,:]
         torchvision.utils.save_image(gtruth, f"{folder}Gtruth{idx}{cc}.tiff")
         x_image=x[cc,:,:]
         toprint=xprint[cc,:,:].to("cpu")
         ntoprint=toprint.numpy()
         cv2.imwrite(f"{folder}x_{idx}{cc}.tiff",ntoprint)
       #torchvision.utils.save_image(toprint, f"{folder}x_{idx}{cc}.tiff")
       #torchvision.utils.save_image(preds, f"{folder}pred_{idx}.tiff")
       #torchvision.utils.save_image(y, f"{folder}{idx}.tiff")
      # torchvision.utils.save_image(x, f"{folder}x_{idx}.tiff")
  model.train()
