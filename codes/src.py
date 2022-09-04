# Name: src.py
# Description: The tools to to train and validate the model
# Author: Behzad Valipour Sh. <behzad.valipour@outlook.com>
# Date: 04.09.2022

'''
MIT License
Copyright (c) 2022 Behzad Valipour Sh.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

import rasterio as rio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window,from_bounds
import geopandas as gpd


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm=False, padding='valid'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BrchNorm = nn.BatchNorm2d(out_channels)
        self.active = nn.LeakyReLU(inplace=True)
        self.norm = norm

    def forward(self,X):
        X = self.conv(X)
        if self.norm is True:
            X = self.BrchNorm(X)
        
        return self.active(X)
    
    
class deConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm=False):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BrchNorm = nn.BatchNorm2d(out_channels)
        self.active = nn.ReLU(inplace=True)
        self.norm = norm
        
    def forward(self,X):
        X = self.deconv(X)
        if self.norm is True:
            X = self.BrchNorm(X)
        
        return self.active(X)
    
    
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, norm=False, padding='valid', use_dropout=False):
        super().__init__()
        
        self.Conv = ConvBlock(in_channels, out_channels, kernel_size, stride, norm=norm, padding=padding)
        self.dropout = nn.Dropout(0.5)
        self.use_dropout = use_dropout
        
    def forward(self,X):
        X = self.Conv(X)
        if self.use_dropout is True:
            X = self.dropout(X)
        return X
    
    
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=4, stride=2, padding=33, norm=True, use_dropout=True):
        super().__init__()
        
        self.deconv = deConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm)
        self.dropout = nn.Dropout(0.5)
        self.use_dropout = use_dropout
        
    def forward(self, X, skip):
        if skip is not None:
            X = torch.cat([X, skip], axis=1)
        X = self.deconv(X)
        if self.use_dropout is True:
            X = self.dropout(X)
        return X
    
    
class SyntheticS2Image(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        """ Encoder """
        
        # Sentinel-2_Before
        self.s2b_conv1  = EncoderBlock(4, 64,padding=33) # Layer_1: 64*64*64
        self.s2b_conv2  = EncoderBlock(64, 128,padding=33) # Layer_2: 128*64*64
        self.s2b_conv3  = EncoderBlock(128, 256, padding=33) # Layer_3: 256*64*64
        self.s2b_conv4  = EncoderBlock(256, 512, padding=33) # Layer_4: 512*64*64
        
        self.s2b_conv5  = EncoderBlock(512, 512, padding=33, use_dropout=True) # Layer_4: 512*64*64
        self.s2b_conv6  = EncoderBlock(512, 512, padding=33, use_dropout=True) # Layer_4: 512*64*64
        
        # Sentinel-2_After
        self.s2a_conv1  = EncoderBlock(4, 64,padding=33) # Layer_1: 64*64*64
        self.s2a_conv2  = EncoderBlock(64, 128,padding=33) # Layer_2: 128*64*64
        self.s2a_conv3  = EncoderBlock(128, 256, padding=33) # Layer_3: 256*64*64
        self.s2a_conv4  = EncoderBlock(256, 512, padding=33) # Layer_4: 512*64*64
        
        self.s2a_conv5  = EncoderBlock(512, 512, padding=33, use_dropout=True) # Layer_4: 512*64*64
        self.s2a_conv6  = EncoderBlock(512, 512, padding=33, use_dropout=True) # Layer_4: 512*64*64
        
        # Sentinel-1
        self.s1_conv1  = EncoderBlock(2, 64,padding=33) # Layer_1: 64*64*64
        self.s1_conv2  = EncoderBlock(64, 128,padding=33) # Layer_2: 128*64*64
        self.s1_conv3  = EncoderBlock(128, 256, padding=33) # Layer_3: 256*64*64
        self.s1_conv4  = EncoderBlock(256, 512, padding=33) # Layer_4: 512*64*64
        
        self.s1_conv5  = EncoderBlock(512, 512, padding=33, use_dropout=True) # Layer_4: 512*64*64
        self.s1_conv6  = EncoderBlock(512, 512, padding=33, use_dropout=True) # Layer_4: 512*64*64

        
        
        """ Decoder """
        self.u6 = DecoderBlock(1536, 512,padding=33)
        self.u7 = DecoderBlock(2048, 512,padding=33)
        self.u8 = DecoderBlock(2048, 256,padding=33)
        self.u9 = DecoderBlock(1024, 128,padding=33)
        self.u10 = DecoderBlock(512, 64,padding=33)
        self.u11 = DecoderBlock(256, 4,padding=33,use_dropout=False)

        
    def forward(self,S2b, S2a, S1):
        S2b1 = self.s2b_conv1(S2b)
        S2b2 = self.s2b_conv2(S2b1)
        S2b3 = self.s2b_conv3(S2b2)
        S2b4 = self.s2b_conv4(S2b3)
        S2b5 = self.s2b_conv5(S2b4)
        S2b6 = self.s2b_conv6(S2b5)
        
        S2a1 = self.s2a_conv1(S2a)
        S2a2 = self.s2a_conv2(S2a1)
        S2a3 = self.s2a_conv3(S2a2)
        S2a4 = self.s2a_conv4(S2a3)
        S2a5 = self.s2a_conv5(S2a4)
        S2a6 = self.s2a_conv6(S2a5)
        
        S11 = self.s1_conv1(S1)
        S12 = self.s1_conv2(S11)
        S13 = self.s1_conv3(S12)
        S14 = self.s1_conv4(S13)
        S15 = self.s1_conv5(S14)
        S16 = self.s1_conv6(S15)

        concat6 = torch.concat([S2b6,S2a6,S16], axis=1)
        U6 = self.u6(concat6,None)
        
        concat5 = torch.concat([S2b5,S2a5,S15], axis=1)
        U7 = self.u7(U6,concat5)
        
        concat4 = torch.concat([S2b4,S2a4,S14], axis=1)
        U8 = self.u8(U7,concat4)
        
        concat3 = torch.concat([S2b3,S2a3,S13], axis=1)
        U9 = self.u9(U8,concat3)
        
        concat2 = torch.concat([S2b2,S2a2,S12], axis=1)
        U10 = self.u10(U9,concat2)
        
        concat1 = torch.concat([S2b1,S2a1,S11], axis=1)
        U11 = self.u11(U10,concat1)
        
        return U11


class LoadImageData(Dataset):
    def __init__(self,GEOJsonFile,S2Before,S2After,S1Now,S2Now, transform=None):
        # data loading
        self.geo = gpd.read_file(GEOJsonFile)
        self.S2N = S2Now
        self.S2B = S2Before
        self.S2A = S2After
        self.S1N = S1Now
        self.n_samples = len(self.geo)
        
        self.transform = transform
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self,idx):
        _S2N = rio.open(self.S2N)
        _S2B = rio.open(self.S2B)
        _S2A = rio.open(self.S2A)
        _S1N = rio.open(self.S1N)
        
        
        minx,miny,maxx,maxy = self.geo.loc[idx, 'geometry'].bounds
        window = from_bounds(minx, miny, maxx, maxy, transform=_S2N.transform)
        
        self.S2B_img = np.moveaxis(_S2B.read(window=window ,resampling=0),0,-1)
        self.S2A_img = np.moveaxis(_S2A.read(window=window ,resampling=0),0,-1)
        self.S1N_img = np.moveaxis(_S1N.read(window=window ,resampling=0),0,-1)
        
        self.S2N_img = np.moveaxis(_S2N.read(window=window ,resampling=0),0,-1)
        
        if self.transform is not None:
            _transformedS2B = self.transform(image = self.S2B_img)
            _transformedS2A = self.transform(image = self.S2A_img)
            _transformedS1N = self.transform(image = self.S1N_img)
            
            _transformedS2N = self.transform(image = self.S2N_img)
            
            self.S2B_img = _transformedS2B["image"]
            self.S2A_img = _transformedS2A["image"]
            self.S1N_img = _transformedS1N["image"]
            
            self.S2N_img = _transformedS2N["image"]
            
            
        return self.S2B_img,self.S2A_img,self.S1N_img,self.S2N_img
    
    
    
class SyntheticS2Model(pl.LightningModule):
    def __init__(self,learning_rate = 0.0001):
        super(SyntheticS2Model,self).__init__()
        
        
        self.model = SyntheticS2Image()
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.save_hyperparameters()
        
    def forward(self,s2b,s2a,s1):
        return self.model(s2b,s2a,s1)
    
    def training_step(self, batch, batch_idx):
        s2b,s2a,s1,y = batch
        
        pred = self(s2b,s2a,s1)
        loss = self.loss(pred,y)
        self.log('Train_Loss', loss, on_epoch=True, on_step=True,prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        s2b,s2a,s1,y = batch
        
        pred = self(s2b,s2a,s1)
        loss = self.loss(pred,y)
        self.log('validation_Loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
        
        
    def test_step(self, batch, batch_idx):
        s2b,s2a,s1,y = batch
        pred = self(s2b,s2a,s1)
        loss = self.loss(pred,y)
        self.log('Test_Loss', loss, on_epoch=True, on_step=True)
        return loss
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer