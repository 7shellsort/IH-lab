import numpy as np
import torch
from torch.autograd import Variable

from helpers.utils import progress_bar
from helpers.utils import format_time

# Train function


def train(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    #如果tune_all为假，只更新最后一层
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    #取水印图像
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        #从哪一对水印图像开始训练
        wm_idx = np.random.randint(len(wminputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        #添加 wmimages 和 targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

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



def train_attack(epoch, net, criterion, optimizer, logfile, loader, device, wmloader=False, attack_loader=False, tune_all=True):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2

    # update only the last layer
    if not tune_all:
        if type(net) is torch.nn.DataParallel:
            net.module.freeze_hidden_layers()
        else:
            net.freeze_hidden_layers()

    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    attinputs, atttargets = [], []
    if attack_loader:
        for att_idx, (attinput, atttarget) in enumerate(attack_loader):
            attinput, atttarget = attinput.to(device), atttarget.to(device)
            attinputs.append(attinput)
            atttargets.append(atttarget)

        # the att_idx to start from
        att_idx = np.random.randint(len(attinputs))
    for batch_idx, (inputs, targets) in enumerate(loader):
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        if attack_loader:
            inputs = torch.cat([inputs, attinputs[(att_idx + batch_idx) % len(attinputs)]], dim=0)
            targets = torch.cat([targets, atttargets[(att_idx + batch_idx) % len(attinputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        usedtime = progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    with open(logfile, 'a') as f:
        f.write('\nEpoch: %d\n' % epoch)
        f.write('Loss: %.3f | Acc: %.3f%% (%d/%d) | time: %s | len(Wm): %d\n'
                % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total, format_time(usedtime), len(loader)))


# Test function
def test(net, criterion, logfile, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.2f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100*(float(correct)/float(total)), correct, total))


    with open(logfile, 'a') as f:
        f.write('Test results:\n')
        f.write('Loss: %.3f | Acc: %.2f%% (%d/%d)\n'
                % (test_loss / (batch_idx + 1), 100*(float(correct)/float(total)), correct, total))
    #返回正确率
    return test_loss / (batch_idx + 1), 100*(float(correct)/float(total))
