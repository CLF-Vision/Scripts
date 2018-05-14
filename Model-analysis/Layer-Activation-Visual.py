# coding: utf-8
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.stats as st
from PIL import Image 
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
import scipy.io as sio
import math
from torchvision import utils
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


im = Image.open('test.png').convert('YCbCr')
im = im.resize((64,64))
y,_,_ = im.split()


plt.imshow(np.asarray(y), cmap='gray')


from model import MSRN
net = MSRN()
net = net.eval()
net = net.cuda()

net.load_state_dict(torch.load('./100.pth')['model'].module.state_dict())


if not os.path.exists('./generated'):
    os.makedirs('./generated')


x = Variable(ToTensor()(y), requires_grad=True).view(1, -1, y.size[1], y.size[0]).cuda()


x = net(x)


def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = utils.make_grid(tensor.cpu(), nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


vistensor(x.cpu().data, ch=1, allkernels=True)

