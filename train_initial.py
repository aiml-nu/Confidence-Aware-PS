#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import io
import torch.utils.data as data_utils
import os
from model import INInet, FeatureResNet
from dataset import Blobby_data
from datetime import datetime
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataString = datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M_%S')

load_model = ''
resume = True
fnet = FeatureResNet()
INInet = INInet(fnet)

### Hyperparameters: You may experiment with different values to achieve improved results
lr = 0.00005   
lr_step = [200, 200]
EPOCH = 100
BATCH_SIZE = 16
NUM_WORKERS = 0
print('BATCH_SIZE = ',BATCH_SIZE)
optimizer = torch.optim.Adam(INInet.parameters(), lr=lr)
start_epoch = 1

if load_model != '':
    checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(load_model, checkpoint['epoch']))
    if type(checkpoint) == type({}):
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint.state_dict()
    INInet.load_state_dict(state_dict, strict=False)
    if resume:
        print('resuming optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(non_blocking=True)
    
INInet = INInet.cuda()

loss_func = nn.MSELoss(reduction = 'sum')
loss_func_mean = nn.L1Loss(reduction = 'sum')

file1 = open('training_set.txt', 'r')
Lines = file1.readlines()
train_set = []
for line in Lines:
    new_item = line[:-1].split(',')
    train_set.append(new_item)
file1 = open('test_set.txt', 'r')
Lines = file1.readlines()
test_set = []
for line in Lines:
    new_item = line[:-1].split(',')
    test_set.append(new_item)
print('==> The number of training set is', len(train_set))
print('==> The number of test set is', len(test_set))

train_data=Blobby_data(dataset=train_set)
train_loader = data_utils.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

test_data=Blobby_data(dataset=test_set)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

Train_root = './Train'+dataString[4:10]+'/'
if not os.path.exists(Train_root):
    os.mkdir(Train_root)
model_root = Train_root+'model/'
log_root = Train_root+'log/'
if not os.path.exists(model_root):
    os.mkdir(model_root)
if not os.path.exists(log_root):
    os.mkdir(log_root)

fileOut=open(log_root+'log'+dataString,'a')
fileOut.write(dataString+'\n'+'Epoch:   Step:    Loss:        Val_Accu :\n')
fileOut.close()
fileOut2 = open(log_root+'validation'+dataString, 'a')
fileOut2.write(dataString+'\n'+'Epoch:    loss:\n')
fileOut2.close()

def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def rev_huber_loss(pred, target):
    diff = (pred - target).abs()
    threshold = diff.max().item()*0.2
    mask = diff.abs() <= threshold
    loss = torch.where(mask, diff, (diff**2 + threshold**2) / (2 * threshold))
    return loss.sum()

for epoch in range(start_epoch,101):
    INInet.train()
    for step, (img,gt1,gt2,gt_mask) in enumerate(train_loader):   
        img = Variable(img).cuda()
        gt1=gt1.unsqueeze(1).float()
        gt1 = Variable(gt1).cuda()
        gt_mask=gt_mask.unsqueeze(1).float()
        gt_mask = Variable(gt_mask).cuda()
        gt2=gt2.float()
        gt2 = Variable(gt2).cuda()
        output = INInet(img)
        loss1 = rev_huber_loss(output[0]*gt_mask, gt2*gt_mask)/(gt_mask.sum()+1e-8)
        loss2 = rev_huber_loss(output[1]*gt_mask, gt1*gt_mask)/(gt_mask.sum()+1e-8) * 2
        loss = loss1+loss2
        optimizer.zero_grad()          
        loss.backward()                 
        optimizer.step()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        if step%12 == 0:
            print('Time: {} epoch: {:03d} step: {:03d} loss: {:.8f} loss_nl: {:.8f}, loss_dp: {:.8f}'.format(time_str, epoch,  step, loss.data.item(),loss1.data.item(),loss2.data.item()))
        fileOut=open(log_root+'log'+dataString,'a')
        fileOut.write(str(epoch)+'   '+str(step)+'   '+str(round(loss.data.item(),8))+'   '+str(round(loss1.data.item(),8))+'   '+str(round(loss2.data.item(),8))+'\n')
        fileOut.close()

    if epoch in lr_step:
        lr = lr * (0.1 ** (lr_step.index(epoch) + 1))
        print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    if epoch%10 == 0:
        save_model(os.path.join(model_root, 'model_{}.pth'.format(epoch)), epoch, INInet, optimizer)
        print('finished saving checkpoints')
    else:
        save_model(os.path.join(model_root, 'model_last.pth'), epoch, INInet, optimizer)
    LOSS_VALIDATION = 0
    LOSS_1 = 0
    LOSS_2 = 0
    loss_1_mean = 0
    loss_2_mean = 0
    INInet.eval()
    with torch.no_grad():
        for step, (img,gt1,gt2,gt_mask) in enumerate(test_loader):
            img = Variable(img).cuda()
            gt1=gt1.unsqueeze(1).float()
            gt1 = Variable(gt1).cuda()
            gt2=gt2.float()
            gt2 = Variable(gt2).cuda()
            gt_mask=gt_mask.unsqueeze(1).float()
            gt_mask = Variable(gt_mask).cuda()
            output = INInet(img) 
            lossv_1 = rev_huber_loss(output[0]*gt_mask, gt2*gt_mask)/(gt_mask.sum()+1e-8)
            lossv_2 = rev_huber_loss(output[1]*gt_mask, gt1*gt_mask)/(gt_mask.sum()+1e-8) * 2
            LOSS_VALIDATION += lossv_1 + lossv_2
            LOSS_1 += lossv_1
            LOSS_2 += lossv_2
            loss_1_mean = loss_1_mean+ loss_func_mean(output[0]*gt_mask, gt2*gt_mask)/(gt_mask.sum()+1e-8)
            loss_2_mean = loss_2_mean+ loss_func_mean(output[1]*gt_mask, gt1*gt_mask)/(gt_mask.sum()+1e-8)
        LOSS_VALIDATION = LOSS_VALIDATION/step
        LOSS_1 = LOSS_1/step
        LOSS_2 = LOSS_2/step
        loss_1_mean = loss_1_mean/step
        loss_2_mean = loss_2_mean/step
        fileOut2 = open(log_root+'validation'+dataString, 'a')
        fileOut2.write(str(epoch)+'   '+str(round(LOSS_VALIDATION.data.item(),8))+'   '+str(round(LOSS_1.data.item(),8))+'   '+str(round(LOSS_2.data.item(),8))+'   '+str(round(loss_1_mean.data.item(),8))+'   '+str(round(loss_2_mean.data.item(),8))+'\n')
        fileOut2.close()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        print('Current time is', time_str)
        print('validation epoch'+ ' ' + str(epoch)+': '+str(round(LOSS_VALIDATION.data.item(),8))+ '   ' +str(round(LOSS_1.data.item(),8))+ '   ' +str(round(LOSS_2.data.item(),8))+ '   ' +str(round(loss_1_mean.data.item(),8))+ '   ' +str(round(loss_2_mean.data.item(),8))+'\n')
