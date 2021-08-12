#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
from torch import optim
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torch.utils.data import DataLoader
from arc_vae.data_loader import Tasks,torchify, Grids, Grids_n
from arc_vae.vae.models import vae
from arc_vae.vae.models import losses
import torch.nn.functional as F
from arc_vae.vae import training
from arc_vae.utils import arc_to_image, visualize_grids


# In[21]:


def get_dataloader(batch_size, eval=False):
   
    tasks = Grids(eval=eval)
    loader = DataLoader(tasks, batch_size=batch_size, drop_last=True)
    
    return loader


# In[30]:


class Model(nn.Module):
    
    def __init__(self):
        
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=11,out_channels=11,kernel_size=5, padding=2)
    
    def forward(self,x):
        
        x = self.conv1(x)
        return x


# In[31]:



def get_optimizer(learning_rate, model):
    
    return optim.Adam(model.parameters(), lr=learning_rate)


# In[32]:


def get_model():
    
    model = Model()
    model.cuda()
    return model

def weight_reset(m):
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m,nn.ConvTranspose2d):
        m.reset_parameters()

def get_optimizer(learning_rate, model):
    
    return optim.Adam(model.parameters(), lr=learning_rate)

def train(model, data_loader, optimizer, epochs):
    
    loss_func = nn.MSELoss()
    
    for epoch in range(epochs):
        
        for data in data_loader:
            inputs,outputs = data
            inputs,outputs = inputs.cuda(),outputs.cuda()


            #outputs = float_long(outputs.cpu())
            optimizer.zero_grad()
            results = model(inputs)

            loss = loss_func(results, outputs)
            loss.backward()
            optimizer.step()
        
        print(f'epoch: {epoch}, loss: {loss}')


# In[33]:


def main():
    epochs = 30
    batch_size = 512
    learning_rate = 0.01
    
    print("getting dataset..")
    train_loader = get_dataloader(batch_size = 512)
    model = get_model()
    optimizer = get_optimizer(learning_rate, model)
    
    model.apply(weight_reset)
    
    print("starting training")
    train(model, train_loader, optimizer, epochs)


# In[34]:


if __name__ == "__main__":
    main()



