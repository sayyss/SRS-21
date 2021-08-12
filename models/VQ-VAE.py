#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[43]:


class Encoder(nn.Module):
    
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4,4,2,1), stride=2):
        
        super(Encoder, self).__init__()
        
        kernel_1,kernel_2,kernel_3,kernel_4 = kernel_size
        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_4, padding=0)
    
    def forward(self,x):
        
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)
        
        x = F.relu(x)
        #print(x.shape)
        y = self.residual_conv_1(x)
        #print(y.shape)
        y = y+x
        
        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y+x
        
        return y


class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super(Decoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes
        
        self.residual_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
        
        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_2, stride, padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_3, stride, padding=0)
        
    def forward(self, x):
        
        y = self.residual_conv_1(x)
        y = y+x
        x = F.relu(y)
        
        y = self.residual_conv_2(x)
        y = y+x
        y = F.relu(y) # 2 2
        
        y = self.strided_t_conv_1(y) #4 4
        
        y = self.strided_t_conv_2(y) # 8 8
        
        return y


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                    torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))
    
    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        
        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        
        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        codebook_loss = F.mse_loss(x.detach(), quantized)
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity

    
class Model(nn.Module):
    
    def __init__(self, Encoder, Codebook, Decoder):
        super(Model, self).__init__()
        
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder
    
    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_quantized)
        
        return x_hat, commitment_loss, codebook_loss, perplexity 


# In[44]:


def get_model(input_dim, output_dim, hidden_dim, n_embeddings):
    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
    codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=hidden_dim)
    decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    model = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to('cuda')
    return model


# In[45]:


def get_dataloader(batch_size, eval=False):
   
    tasks = Grids_n(eval=eval)
    loader = DataLoader(tasks, batch_size=batch_size, drop_last=True)
    
    return loader


# In[50]:



def weight_reset(m):
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m,nn.ConvTranspose2d):
        m.reset_parameters()

def get_optimizer(learning_rate, model):
    
    return optim.Adam(model.parameters(), lr=learning_rate)


# In[52]:


loss_func = nn.MSELoss()

def train(model, data_loader, optimizer, epochs):
    
    for epoch in range(epochs):
        
        for data in data_loader:
            data = data.cuda()
            optimizer.zero_grad()
            
            out, commitment_loss, codebook_loss, perplexity = model(data)
            
            recon_loss = loss_func(out,data)
            loss = recon_loss + commitment_loss + codebook_loss
            
            loss.backward()
            optimizer.step()
        
        print(f'epoch: {epoch}, recon_loss: {recon_loss.item()}, perplexity: {perplexity.item()},         commitment_loss: {commitment_loss.item()}, codebook_loss: {codebook_loss}, overall_loss: {loss}')


# In[53]:



def main():
    batch_size = 512
    img_size = (10,10)
    learning_rate = 2e-3
    epochs = 1000
    
    input_dim = 11
    output_dim = 11
    hidden_dim = 256
    n_embeddings= 768
    
    print("getting dataset..")
    train_loader = get_dataloader(batch_size=batch_size)
    
    
    model = get_model(input_dim,output_dim, hidden_dim, n_embeddings)
    optimizer = get_optimizer(learning_rate, model)
    model.apply(weight_reset)
    
    print("starting training")
    train(model, train_loader, optimizer, epochs)
    torch.save(model.state_dict(), 'VQ-VAE-3.pt')
    # Evaluation
    print("Testing model")
    eval_loader = get_dataloader(batch_size=10)
    test(eval_loader,model)
# In[54]:


if __name__ == "__main__":
    main()




