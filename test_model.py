#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arc_vae.utils import arc_to_image, visualize_grids


def show(count,img):
    npimg = img.numpy()
    plt.figure(figsize = (30,30))
    plt.imshow(np.transpose(npimg,(1,2,0)), interpolation="nearest")
    plt.savefig(f'test_images/{count}.png')
    
def test(data_loader, model):
    
    model.eval()
    print("testing model...")
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        
    count = 0
    for data in data_loader:
        
        inputs = data
        inputs = inputs.cuda()
    
        out = model(inputs)[0]
        show(count,visualize_grids(inputs))
        show(str(count)+"_out", visualize_grids(out))
        break