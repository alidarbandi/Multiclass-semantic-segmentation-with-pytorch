# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 15:02:38 2022

@author: sem
"""
import utils2
import Unet_tutorial
import carvana_data
#!pip install wandb
#!pip install albumentations==0.4.6

import torch
import albumentations as A
#import wandb
import albumentations
import torch.nn as nn
import torch.optim as optim
from Unet_tutorial import Unet
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils2 import get_loaders, check_acc, save_pred, save, load, save_best
import torchvision.transforms.functional as TF
import torchvision.transforms
import statistics
import numpy
#wandb.init()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
name=torch.cuda.get_device_name(device)
print(f'our cuda device is {name}')
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs,
#     )

### Hyper paramters
lrate=0.0005


epoch=1
batch=5
img_w=672 #672 1344
img_h=320 #320  656
need_load=True
#num_workers=1
#pin_memory=True



path='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/model.pth'
best_path='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/best_model.pth'

image_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/image/'
mask_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/mask/'

val_image_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/valid_image/'
val_mask_dir='Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/valid_mask/'

loss_list=[]


def train_fn(loader, model, optimizer, loss_fn,scaler):
    loop=tqdm(loader)
    
    transform_normal=torchvision.transforms.Compose([
         torchvision.transforms.Normalize(179,54)                                        

       ])
    
    for batch_idx, (raw_data, target) in enumerate(loop):
       # un_data=raw_data.float().unsqueeze(1) #.to(device)
        un_data=raw_data.float()
        target=target.long().to(device)
        data=transform_normal(un_data).unsqueeze(1).to(device)
      #  numpy_data=data.numpy()
      #  mean=numpy.mean(numpy_data)
      #  std=numpy.std(numpy_data)
      #  print(f'mean of image is {mean} with STD of {std}')
      #  print(torch.max(data))
      # data=data.to(device)
       # print(f'data shape is {data.shape}')
       # print(f'mask shape is {target.shape}')
        
       
       # target=torch.sigmoid(target).to(device)

      #  target=target.float().unsqueeze(1).to(device)
       
        #### forward pass
        with torch.cuda.amp.autocast():
          #model.forward(tensor.half())
          prediction=model(data)
          loss=loss_fn(prediction,target)
       ### backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
       # scaler.update()
        loss_list.append(loss.detach().cpu().item())
        loop.set_postfix(loss=loss.item())
        
def main():
    train_transform=A.Compose(
        [
            A.Resize(height=img_h,width=img_w),
            A.Rotate(limit=30,p=0.2),
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.1),
  #          A.Normalize(
   #             mean=[0],
              #  std=[1],
               # max_pixel_value=255,
    #            ),
          #  ToTensorV2(),
            
        
        ])

    val_transform=A.Compose(
        [
            A.Resize(height=img_h,width=img_w),
              
        ])
          

    model=Unet(in_channel=1, out_channel=3).to(device)
   # loss_fn=nn.BCEWithLogitsLoss()
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lrate)
    
    train_loader, val_loader = get_loaders(
    #train_loader = get_loaders(
       image_dir,
       mask_dir,
       val_image_dir,
       val_mask_dir,
       batch,
       train_transform,
       val_transform,
 #      num_workers=num_workers,
  #     pin_memory=pin_memory,
   )
   
  
    scaler = torch.cuda.amp.GradScaler()

    if need_load:
        best_score=load(path, model,optimizer)
    else:
      best_score=0
    
    epoc_loss=[]
    for items in range(epoch):
       train_fn(train_loader, model, optimizer, loss_fn,scaler)
       print(f'running epoch # {items}')
       epoc_loss.append(round(statistics.mean(loss_list),2))
       print(f'list of loss is {epoc_loss}')
       check_acc(val_loader,model,device="cuda")
   #    cum_score=check_acc(val_loader,model,device="cuda")
    #   current_dice=cum_score/len(val_loader)
   #    print(f'dice score is {current_dice:.3f}')
   #    if current_dice>best_score:
    #     best_score=current_dice
     #    checkpoint = {
      #      "state_dict": model.state_dict(),
       #     "optimizer":optimizer.state_dict(),
        #    "score": best_score,
         #   }
        # save_best(checkpoint,best_path)
       
       
       save_pred(val_loader,model,folder="Z:/Images/Ali/Human Liver/Nov 17/Jan 20 Unet_multi_Hela/Unet_multi/save_preds/")
       checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "score": best_score,
        }
       save(checkpoint,path)

if __name__=="__main__":
   main()      
      

