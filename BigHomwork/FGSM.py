from __future__ import print_function

import argparse
import os
import time
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from helpers.loaders import *
from helpers.utils import progress_bar
from helpers.utils import format_time
from helpers.utils import adjust_learning_rate

parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=2, type=int, help='the wm batch size')
parser.add_argument('--max_epochs', default=60, type=int, help='the maximum number of epochs')
parser.add_argument('--lradj', default=20, type=int, help='multiple the lr by 0.1 every n epochs')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--save_model', default='FGSM_model.t7', help='model name')
parser.add_argument('--load_path', default='./checkpoint/model.t7', help='the path to the pre-trained model, to be used with resume flag')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wmtrain', '-wmt', action='store_true', help='train with wms?')
parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='FGSM_Attack', help='the exp name')
parser.add_argument('--FGSM_attack', '-FGSM', action='store_true', help='start FGSM attack?')

args = parser.parse_args()

def fgsm_attack(image, epsilon, data_grad):
    #收集数据梯度的逐元素符号
    sign_data_grad = data_grad.sign()

    #通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon*sign_data_grad

    #添加裁剪以保持 [0,1] 范围
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def attack(model, tensor_image, label, epsilon=8/255):
    #attack 函数用来生成对抗性样本，'tensor_image'是输入图像张量，'label'是图像的正确标签，'epsilon'是扰动的大小

    #把图像张量转换成变量，Variable:类似于一个tensor的升级版，里面包含了requires_grad,grad_fn,voliate
    image = torch.autograd.Variable(tensor_image, requires_grad=True)
    output = model(image)
    
    loss = F.nll_loss(output, label)    #计算损失
    model.zero_grad()     
    loss.backward()
    
    #收集数据
    data_grad = image.grad.data 
    #调用 FGSM 攻击
    perturbed_image = fgsm_attack(image, epsilon, data_grad)
    #将扰动图像转换回张量
    perturbed_tensor = perturbed_image.data
    return perturbed_tensor

# Test function
def test(net, criterion, logfile, loader, device, Attack = False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        if Attack:
            inputs = attack(net, inputs, targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # return the acc.
    return 100. * correct / total

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logfile = os.path.join(LOG_DIR, 'log_' + str(args.runname) + '.txt')
    confgfile = os.path.join(LOG_DIR, 'conf_' + str(args.runname) + '.txt')

    term_width = shutil.get_terminal_size().columns
    print(term_width)
    print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))

    print('==> Resuming from module.t7..')
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    print('Loading watermark images')
    wmloader = getwmloader(args.wm_path, args.wm_batch_size, args.wm_lbl)

    print("the original acc:")
    test(net, criterion, logfile, wmloader, device)
    print("After the FGSM attack:")
    test(net, criterion, logfile, wmloader, device, Attack=True)
    
