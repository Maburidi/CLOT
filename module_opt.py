
import os
import argparse
import torch
import torchvision
import numpy as np
from torch.utils import data
import copy
import argparse       
import cv2          
import torch.nn as nn            
from torch.nn.functional import normalize       
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1
import math     
import torchvision.datasets as datasets

from scipy.special import logsumexp
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MovingAverage():
    def __init__(self, intertia=0.9):
        self.intertia = intertia
        self.reset()

    def reset(self):
        self.avg = 0.

    def update(self, val):
        self.avg = self.intertia * self.avg + (1 - self.intertia) * val



def optimize_epoch(optimizer,model, loader, epoch, lr_schedule, lr, lamda ,device,K,hc,dtype):
    loss_value = AverageMeter() 
    lr = lr_schedule(epoch)
    XE = torch.nn.CrossEntropyLoss() 
    for pg in optimizer.param_groups:
        pg['lr'] = lr                 
    uu=0
    
    for iter, (data1, data2, selected, label) in enumerate(loader):                     

        niter = epoch * len(loader) + iter

        with torch.no_grad():                       
            if torch.cuda.is_available():               
                if torch.cuda.device_count() ==1: 
                    data1 = data1.to(torch.device('cuda:0'))
                    data2 = data2.to(torch.device('cuda:0'))
                    final_i, final_j, c_i, c_j = model(data1,data2)
                    Q = gpu_sk(optimizer, model=model, dataloader=data_loader, K=K,hc=hc,lamda=lamda, c_i = c_i, c_j = c_j, selected= selected) 

                elif torch.cuda.device_count() > 1:
                    print("Code is designed to use only one GPU") 
            else: 
                print("Code is designed to use only one GPU") 
                #Q = cpu_sk(optimizer, model=model, loader=data_loader,dtype=dtype, K=K, dev=device,hc=hc, Q=L)
   
        data1 = data1.to("cuda:0")
        data2 = data2.to("cuda:0")

        mass = data1.size(0)
        
        #################### train CNN #########################################
        loss = XE(final_i, Q)

        optimizer.zero_grad()                  
        loss.backward()                        
        optimizer.step()                        
        loss_value.update(loss.item(), mass)     
        
        print(niter, " Loss: {0:.3f}".format(loss.item()), flush=True)
    return {'loss': loss_value.avg}                
    
def gpu_sk(optimizer,model,dataloader ,K,hc ,lamda,c_i,c_j,selected, dtype=torch.float64):
    #N = len(dataloader.dataset)
    N = dataloader.batch_size
 

    PS = torch.empty(N, K, device='cuda:0', dtype=dtype)
    batch_time = MovingAverage(intertia=0.9)

    softmax = torch.nn.Softmax(dim=1).to('cuda:0')
    now = time.time() 
    uuu=0 

    #for batch_idx, (x_i, x_j, selected, label) in enumerate(dataloader):
              
    #x_i = x_i.to(torch.device('cuda:0'))
    #x_j = x_j.to(torch.device('cuda:0'))
    #z_i, z_j, c_i, c_j = model(x_i, x_j)
    
      
    PS = 0.5*(c_i.detach().to(dtype) + c_j.detach().to(dtype))
      
    print("The matrix P is obtained in {0:.3f} min".format((time.time() - now) / 60.), flush=True)
    
    # 2. Solve label assignment via sinkhorn-knopp:
    
    tt = time.time() 
    r = torch.ones((K, 1), device='cuda:0', dtype=dtype) / K 
    c = torch.ones((N, 1), device='cuda:0', dtype=dtype) / N
    
    ones = torch.ones(N, device='cuda:0', dtype=dtype)
    inv_K = 1. / K 
    inv_N = 1. / N
    PS = torch.transpose(PS, 0, 1)      # K X N 

    PS= PS.pow_(lamda) 
    
    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (torch.matmul(PS, c))
        c_new  = inv_N / torch.transpose(torch.matmul(torch.transpose(r, 0, 1), PS),0,1) 
        if _counter % 10 == 0:
            err = torch.sum(torch.abs((c.squeeze() / c_new.squeeze()) - ones)).cpu().item()
        c = c_new
        _counter += 1
    
    #print(c)

    torch.mul(PS, c[0,:].to('cuda:0'), out = PS)
    PS = torch.transpose(PS,0,1)
    PS2 = torch.mul(PS, r[0,:].to('cuda:0'))    
    argmaxes = torch.empty(N, dtype=torch.int64, device='cuda:0')
    amax = torch.argmax(PS2, 1) 
    argmaxes.copy_(amax)
    
    #Q[0] = argmaxes 
    print("The label matrix Q is obtained in {0:.3f} min".format((time.time() - tt) / 60.), flush=True)
    return argmaxes 

def cpu_sk(optimizer,model, loader, dtype, K, dev,hc,Q ):
    N = len(loader.dataset)
    PS = np.zeros((N, K), dtype=dtype)
    u =0
    outs = [K]*hc  
    nh=0
    
    now = time.time()    
    time.time()
    #batch_time = MovingAverage(intertia=0.9)

    # 1. Obtain the matrix P 

    for batch_idx, (x_i, x_j, selected, label) in enumerate(loader):
        x_i = x_i.to('cpu')
        x_j = x_j.to('cpu')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        PS[selected, :] = c_j.detach().cpu().numpy().astype(dtype) 
                         
        #if u == 1:                           
         # break                             
        #u = u+1                                
                                       
    print("The matrix P is obtained in {0:.3f} min".format((time.time() - now) / 60.), flush=True)
    tt = time.time() 
    # 2. Solve label assignment via sinkhorn-knopp:
    PS = PS.T           # now it is K x N
    r = np.ones((outs[nh], 1), dtype=dtype) / outs[nh]
    c = np.ones((N, 1), dtype=dtype) / N
    
    inv_K = dtype(1./outs[nh])
    inv_N = dtype(1./N)

    err = 1e6
    _counter = 0
    while err > 1e-1:
        r = inv_K / (PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1

    PS *= np.squeeze(c) 
    PS = PS.T
    PS *= np.squeeze(r)
    PS = PS.T
    argmaxes = np.nanargmax(PS, 0)            # size N
    newL = torch.LongTensor(argmaxes)  
    Q[nh] = newL.to(dev)
    print("The label matrix Q is obtained in {0:.3f} min".format((time.time() - tt) / 60.), flush=True)
     
    return Q

                          



