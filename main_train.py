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


def save_model(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label')

    # TRAIN AND DATA OPTIONS
    parser.add_argument('--batch-size', default=50, type=int, help='batch size (default: 256)')
    parser.add_argument("--image_size", type=int, default=224, nargs="+", help="crop size of an image")
    parser.add_argument("--epochs", default=1000, type=int, help="number of total epochs to run")         
    parser.add_argument('--dataset', default='CIFAR-10', type=str, help='CIFAR-10, CIFAR-100, ImageNet')
    parser.add_argument("--dataset_dir", type=str, default="/content/drive/MyDrive/Colab_Notebooks/self_super_project/train",help="dir for training data")
    
    # MODEL               
    parser.add_argument('--resnet', default='ResNet50', type=str, help='ResNet50, ResNet34')
    parser.add_argument("--feature_dim", default=128, type=int, help="feature dimension")                             
    parser.add_argument("--model_path", type=str, default="/content/drive/MyDrive/Colab_Notebooks/self_super_project/self-label-default/checkpoints",help="experiment dump path for checkpoints and to save model")
    parser.add_argument("--reload", default=True, type=bool, help="True if resume training, False if start from scratch")                             
    parser.add_argument('--hc', default=1, type=int, help='number of heads (default: 1)')
    parser.add_argument('--class-num', default=3, type=int, help='Aimed number of clsuters')
    parser.add_argument('--start-epoch', default=0, type=int, help='start-epoch')


    # CHECKPOINTS         
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--workers", default=2, type=int, help="""number of processes: it is set automatically and should not be passed as argument""")

    # LOSS OPTIONS 
    parser.add_argument("--instance-temperature", default=0.5, type=float, help="temperature parameter in training loss")
    parser.add_argument("--cluster-temperature", default=1.0, type=float, help="temperature cluster parameter in training loss")
    parser.add_argument("--weight-decay", default=0., type=float, help="decaying weight")
    parser.add_argument('--learning-rate', default=0.0003, type=float, help='initial learning rate (default: 0.05)')
    parser.add_argument('--momentum', default=0.9, type=float, help='initial learning rate (default: 0.05)')

    return parser.parse_args(args=[])

if __name__ == "__main__":

    ################################# Inputs ###################################
    args = get_parser()
    print("We have the following parameters!")
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run will be done on {device}")

    ########################### Set the data loader ############################
    train_dataset = cifar_10_data( data_path = args.dataset_dir)     
    data_loader = torch.utils.data.DataLoader(                    
        train_dataset,                                          
        batch_size=args.batch_size,         
        shuffle=True,               
        num_workers=1, 
        pin_memory=True,
        drop_last=False 
    )                   

    print(f"Number of data loaded points = {len(data_loader.dataset)}")        

    ############################# Set the model ################################ 
    res = get_resnet(args.resnet)
    model = Network(res, args.feature_dim, args.class_num)
    model = model.to(device)


    ############################################################################
    ###################           Optimization           #######################
    ############################################################################

    #===================== Set the optimizer and the loss ======================
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    weight_decay=args.weight_decay,
                                    momentum=args.momentum,
                                    lr=args.learning_rate)
    loss_device = torch.device(device)

    #===================== initilize the labels randomally =====================

    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1


    N = len(data_loader.dataset) 
    args.hc = 1   
    K =args.class_num      #number of classes
    outs = [K]*args.hc   

    if args.start_epoch == 0:
        L = np.zeros((args.hc, N), dtype=np.int32)
        for nh in range(args.hc):
            for _i in range(N):
                L[nh, _i] = _i % outs[nh]
            L[nh] = np.random.permutation(L[nh])
        L = torch.LongTensor(L).to(device)

   #====================== optimizing for each epoch ===========================
    lowest_loss = 1e9
    epoch = args.start_epoch
    lrdrop = 150
    lamda = 25 
    lr_schedule = lambda epoch: ((epoch < 350) * (args.learning_rate * (0.1 ** (epoch // lrdrop)))    
                                          + (epoch >= 350) * args.learning_rate * 0.1 ** 3)            
    
    if torch.cuda.is_available():                                                        
        dtype = torch.float64
    else:
        dtype = np.float64

    while epoch < (args.epochs+1): 
        print(f"===========>>>> Start epoch number: {epoch} <<<<================= ")
        time1 = time.time()
        m = optimize_epoch(optimizer,model =model, loader=data_loader, epoch=epoch,lr_schedule= lr_schedule, lr=args.learning_rate, device =device, lamda=lamda, K=K, hc=args.hc, dtype = dtype)
        loss_epoch =  m['loss']
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}\t took: {(time.time() - time1) / 60.} min")
        if epoch % 10 == 0:
            save_model(args.model_path, model, optimizer, epoch)
        epoch += 1
    #save_model(model_path, model, optimizer,epochs)

