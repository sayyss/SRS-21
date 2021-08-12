#!/usr/bin/env python
# coding: utf-8

# In[48]:


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
from test_model import test

# In[49]:


def get_dataloader(batch_size, eval=False):
    
    tasks = Grids_n(eval=eval)
    loader = DataLoader(tasks, batch_size=batch_size, drop_last=True)
    
    return loader


# In[50]:


class VAE(nn.Module):
    def __init__(self,imgChannels=11, zDim=64):
        super(VAE, self).__init__()

        self.encConv1 = nn.Conv2d(imgChannels, 50, 3)
        self.encConv2 = nn.Conv2d(50, 100, 3)
        self.encFC1 = nn.Linear(128*2*2, zDim)
        self.encFC2 = nn.Linear(128*2*2, zDim)

        self.decFC1 = nn.Linear(zDim, 128*2*2)
        self.decConv1 = nn.ConvTranspose2d(36, 16, 1)
        self.decConv2 = nn.ConvTranspose2d(16, 11, 1)

    def encoder(self, x):
        
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        
        x = x.view(-1, x.size()[0])
        
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        
        return mu, logVar

    def reparameterize(self, mu, logVar):

        
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        
        
        x = F.relu(self.decFC1(z))
        batch_size = x.shape[1]
        x = x.view(batch_size,-1,10,10)
        
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
       
        return x

    def forward(self, x):

        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        
        out = self.decoder(z)
        return out, mu, logVar


# In[51]:


def get_model():
    
    model = VAE()
    model.cuda()
    return model

def weight_reset(m):
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m,nn.ConvTranspose2d):
        m.reset_parameters()

def get_optimizer(learning_rate, model):
    
    return optim.Adam(model.parameters(), lr=learning_rate)

def train(model, data_loader, optimizer, epochs):
    
    for epoch in range(epochs):
        
        for data in data_loader:
            inputs = data
            
            inputs = inputs.cuda()
            
            out,mu,logVar = model(inputs)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, inputs) + kl_divergence
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'epoch: {epoch}, loss: {loss}')


# In[52]:



def main():
    batch_size = 512
    epochs = 1
    
    print("getting dataset..")
    train_loader = get_dataloader(batch_size=batch_size)
    model = get_model()
    optimizer = get_optimizer(1e-3, model)
    
    model.apply(weight_reset)
    
    print("starting training")
    train(model, train_loader, optimizer, epochs)
    torch.save(model.state_dict(), 'VAE-3.pt')
    # Evaluation
    print("Testing model")
    eval_loader = get_dataloader(batch_size=10)
    test(eval_loader,model)

# In[53]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




