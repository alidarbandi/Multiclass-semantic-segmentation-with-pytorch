# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:51:26 2021

@author: alida
"""

import torch
from torch import nn
#from torchinfo import summary
import torchvision.transforms.functional as TF

class Doubleconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Doubleconv,self).__init__()
        self.conv=nn.Sequential(
                 nn.Conv2d(in_channel,out_channel,3,1,1,bias=False),
                 nn.BatchNorm2d(out_channel),
                 nn.ReLU(),
                 nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
                 nn.BatchNorm2d(out_channel),
                 nn.ReLU()               
            )
    def forward(self,x):
        return self.conv(x)
    

class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, features=[64,128,256,512]):
        super(Unet,self).__init__()
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
        self.pool=nn.MaxPool2d(2, stride=2)
        
        #### The downs 
        for feature in features:
            self.downs.append(Doubleconv(in_channel,feature))
            in_channel=feature
        ### the Ups 
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(2*feature, feature,kernel_size=2, stride=2))
            self.ups.append(Doubleconv(2*feature, feature))
        self.bottleneck=Doubleconv(features[-1], 2*features[-1])
        self.finalconv=nn.Conv2d(features[0],out_channel, kernel_size=1)
        
    def forward(self,x):
        skip_connection=[]
        
        for down in self.downs:
          x=down(x)
         # print(f'going down, the input size is: {x.shape[1:]}')
          skip_connection.append(x)
          x=self.pool(x)
        x=self.bottleneck(x)
        #print(f'At bottleneck, the input size is: {x.shape[1:]}')
        skip_connection=skip_connection[::-1]
        for idx in range(0,8,2):
            x=self.ups[idx](x)
            #print(f'Going up, the input size is: {x.shape[1:]}')
            skip=skip_connection[idx//2]
            if x.shape != skip.shape:
               x=TF.resize(x,size=skip.shape[2:])
            concat_skip=torch.cat((skip,x),dim=1) #### dimensions are batch=0, channel=1, height=2, width=3
            x=self.ups[idx+1](concat_skip)
        return self.finalconv(x)


    
#report=summary(model, input_size=(1, 1, 1353, 667))
#print(report)      

# inputs=torch.randn( 1,1,160,160)
# model=Unet()
# outputs=model(inputs)
# print(inputs.shape)
# print(outputs.shape)           
        
        
        
        
        
        
        
        
        
        
                    
                        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        