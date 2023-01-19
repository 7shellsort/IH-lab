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
from trainer import test

parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--dataset', default='cifar10', help='the dataset to train on [cifar10]')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')
parser.add_argument('--wm_batch_size', default=2, type=int, help='the wm batch size')
parser.add_argument('--max_epochs', default=20, type=int, help='the maximum number of epochs')
parser.add_argument('--lradj', default=20, type=int, help='multiple the lr by 0.1 every n epochs')
parser.add_argument('--save_dir', default='./checkpoint/', help='the path to the model dir')
parser.add_argument('--save_model', default='DataPoison_model.t7', help='model name')
parser.add_argument('--load_path', default='./checkpoint/model.t7', help='the path to the pre-trained model, to be used with resume flag')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--wmtrain', '-wmt', action='store_true', help='train with wms?')
parser.add_argument('--log_dir', default='./log', help='the path the log dir')
parser.add_argument('--runname', default='DataPoison_Attack', help='the exp name')
parser.add_argument('--DataPoison_attack', '-DPoison', action='store_true', help='start data poison attack?')

args = parser.parse_args()

def datapoison(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    #只更新最后一层
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        #定义毒药张量
        poisoning_tensor = 0.06 * torch.randn(3, 32, 32)
        inputs = inputs + poisoning_tensor

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        usedtime = progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100*(float(correct)/float(total)), correct, total))

    with open(logfile, 'a') as f:
        f.write('\nEpoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.2f%% (%d/%d) | time: %s | len(Wm): %d\n'
                % (train_loss / (batch_idx + 1), 100*(float(correct)/float(total)), correct, total, format_time(usedtime), len(loader)))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)
    logfile = os.path.join(LOG_DIR, 'log_' + str(args.runname) + '.txt')
    confgfile = os.path.join(LOG_DIR, 'conf_' + str(args.runname) + '.txt')

    # save configuration parameters
    with open(confgfile, 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))

    trainloader, testloader, n_classes = getdataloader(
        args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

    wmloader = getwmloader(args.wm_path, args.wm_batch_size, args.wm_lbl)

    print('==> Resuming from module.t7..')
    checkpoint = torch.load(args.load_path)
    net = checkpoint['net']
    acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

    net = net.to(device)
    # support cuda
    if device == 'cuda':
        print('Using CUDA')
        print('Parallel training on {0} GPUs.'.format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # start training
    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        # adjust learning rate
        adjust_learning_rate(args.lr, optimizer, epoch, args.lradj)

        datapoison(epoch, net, criterion, optimizer, logfile, trainloader, device)

        print("Test acc:")
        acc = test(net, criterion, logfile, testloader, device)

        print("WM acc:")
        test(net, criterion, logfile, wmloader, device)

        print('Saving..')
        state = {
            'net': net.module if device == 'cuda' else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)
        torch.save(state, os.path.join(args.save_dir, args.save_model))

if __name__ == "__main__":
    main()
