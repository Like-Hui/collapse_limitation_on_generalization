import argparse
import os
import sys
import shutil
import time

import pickle
import itertools
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data import random_split
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor

import torchvision.models as models

from feature_ana import feature_analysis

from resnet import ResNet18
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--num_classes', default=10, type=int,
                    help='class number')
parser.add_argument('--max_step', default=80001, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='SGD', type=str,
                    help='optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--loss_type', default='CE', type=str,
                    help='name of experiment')
parser.add_argument('--new_loss', action='store_true',
                            help='sc loss or not')
parser.add_argument('--weighted', default=0, type=int, help='reweight the loss at true label by ?')
parser.add_argument('--rescale_factor', default=1, type=int, help='rescale the one hot vector by how much?')
parser.add_argument('--rescale', default=1, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--subset_num', default=3500, type=int,
                    help='subset sample size')
parser.add_argument('--save_fig_dir', default='svhn_ce', type=str,
                    help='name of experiment')

parser.set_defaults(augment=True)

class Graphs:
    def __init__(self):
        self.accuracy = []
        self.loss = []
        self.reg_loss = []

        # NC1
        self.Sw_invSb = []

        # NC2
        self.norm_M_CoV = []
        self.norm_W_CoV = []
        self.cos_M = []
        self.cos_W = []

        # NC3
        self.W_M_dist = []

        # NC4
        self.NCC_mismatch = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []


train_step = 0
best_prec1 = 0
graphs = Graphs()
graphs_test = Graphs()

global args
args = parser.parse_args()

if args.tensorboard: configure("runs/%s"%(args.name))

step_list = range(0, args.max_step + 1, 8000)

print(args)
t = time.perf_counter()
# Data loading code

train_dataset = SVHN(
        root='data/', split='train', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )

test_dataset = SVHN(
        root='data/', split='test', download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )

kwargs = {'num_workers': 1, 'pin_memory': True}

print('data shape: ', len(train_dataset), len(test_dataset))
print('data label: ', np.min(train_dataset.labels))
mask = torch.ones(len(train_dataset))
indices_0 = torch.nonzero(torch.tensor((train_dataset.labels == 0)) * mask)
indices_1 = torch.nonzero(torch.tensor((train_dataset.labels == 1)) * mask)
indices_2 = torch.nonzero(torch.tensor((train_dataset.labels == 2)) * mask)
indices_3 = torch.nonzero(torch.tensor((train_dataset.labels == 3)) * mask)
indices_4 = torch.nonzero(torch.tensor((train_dataset.labels == 4)) * mask)
indices_5 = torch.nonzero(torch.tensor((train_dataset.labels == 5)) * mask)
indices_6 = torch.nonzero(torch.tensor((train_dataset.labels == 6)) * mask)
indices_7 = torch.nonzero(torch.tensor((train_dataset.labels == 7)) * mask)
indices_8 = torch.nonzero(torch.tensor((train_dataset.labels == 8)) * mask)
indices_9 = torch.nonzero(torch.tensor((train_dataset.labels == 9)) * mask)

print('train size of different class: ', len(indices_0), len(indices_1), len(indices_2), len(indices_3), len(indices_4), len(indices_5), len(indices_6), len(indices_7), len(indices_8), len(indices_9))

indices_0 = indices_0[:args.subset_num]
indices_1 = indices_1[:args.subset_num]
indices_2 = indices_2[:args.subset_num]
indices_3 = indices_3[:args.subset_num]
indices_4 = indices_4[:args.subset_num]
indices_5 = indices_5[:args.subset_num]
indices_6 = indices_6[:args.subset_num]
indices_7 = indices_7[:args.subset_num]
indices_8 = indices_8[:args.subset_num]
indices_9 = indices_9[:args.subset_num]
indices = torch.cat((indices_0, indices_1))
indices = torch.cat((indices, indices_2))
indices = torch.cat((indices, indices_3))
indices = torch.cat((indices, indices_4))
indices = torch.cat((indices, indices_5))
indices = torch.cat((indices, indices_6))
indices = torch.cat((indices, indices_7))
indices = torch.cat((indices, indices_8))
indices = torch.cat((indices, indices_9))

new_train_dataset = torch.utils.data.Subset(train_dataset, indices)

mask_test = torch.ones(len(test_dataset))
te_indices_0 = torch.nonzero(torch.tensor((test_dataset.labels == 0)) * mask_test)
te_indices_1 = torch.nonzero(torch.tensor((test_dataset.labels == 1)) * mask_test)
te_indices_2 = torch.nonzero(torch.tensor((test_dataset.labels == 2)) * mask_test)
te_indices_3 = torch.nonzero(torch.tensor((test_dataset.labels == 3)) * mask_test)
te_indices_4 = torch.nonzero(torch.tensor((test_dataset.labels == 4)) * mask_test)
te_indices_5 = torch.nonzero(torch.tensor((test_dataset.labels == 5)) * mask_test)
te_indices_6 = torch.nonzero(torch.tensor((test_dataset.labels == 6)) * mask_test)
te_indices_7 = torch.nonzero(torch.tensor((test_dataset.labels == 7)) * mask_test)
te_indices_8 = torch.nonzero(torch.tensor((test_dataset.labels == 8)) * mask_test)
te_indices_9 = torch.nonzero(torch.tensor((test_dataset.labels == 9)) * mask_test)

print('test size of different class: ', len(te_indices_0), len(te_indices_1), len(te_indices_2), len(te_indices_3), len(te_indices_4), len(te_indices_5), len(te_indices_6), len(te_indices_7), len(te_indices_8), len(te_indices_9))

te_indices_0 = te_indices_0[:int(args.subset_num*0.2)]
te_indices_1 = te_indices_1[:int(args.subset_num*0.2)]
te_indices_2 = te_indices_2[:int(args.subset_num*0.2)]
te_indices_3 = te_indices_3[:int(args.subset_num*0.2)]
te_indices_4 = te_indices_4[:int(args.subset_num*0.2)]
te_indices_5 = te_indices_5[:int(args.subset_num*0.2)]
te_indices_6 = te_indices_6[:int(args.subset_num*0.2)]
te_indices_7 = te_indices_7[:int(args.subset_num*0.2)]
te_indices_8 = te_indices_8[:int(args.subset_num*0.2)]
te_indices_9 = te_indices_9[:int(args.subset_num*0.2)]

te_indices = torch.cat((te_indices_0, te_indices_1))
te_indices = torch.cat((te_indices, te_indices_2))
te_indices = torch.cat((te_indices, te_indices_3))
te_indices = torch.cat((te_indices, te_indices_4))
te_indices = torch.cat((te_indices, te_indices_5))
te_indices = torch.cat((te_indices, te_indices_6))
te_indices = torch.cat((te_indices, te_indices_7))
te_indices = torch.cat((te_indices, te_indices_8))
te_indices = torch.cat((te_indices, te_indices_9))

new_test_dataset = torch.utils.data.Subset(test_dataset, te_indices)

train_loader = torch.utils.data.DataLoader(
    new_train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(
    new_test_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = models.vgg11_bn(pretrained=False, num_classes=10)

for name, para in model.named_modules():
    print(name, para)

# get the number of model parameters
print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

model = model.cuda()
# for name, para in model.named_modules():
#     print(name, para)
#
# sys.exit()

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
criterion_summed = nn.CrossEntropyLoss(reduction='sum')

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum, nesterov = args.nesterov,
                            weight_decay=args.weight_decay)

# cosine learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step)

class fc_features:
    pass

def fc_hook(self, input, output):
    fc_features.value = input[0].clone()

# register hook that saves last-layer input into features

class classifier:
    pass

def classifier_hook(self, input, output):
    classifier.value = input[0].clone()

fc_layer = model.classifier[6]
fc_layer.register_forward_hook(fc_hook)

classifier1 = model.classifier[6]
classifier1.register_forward_hook(classifier_hook)

cur_epochs = []
graphs = Graphs()
graphs_test = Graphs()

def validate(val_loader):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    test_losses = AverageMeter()
    test_top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            output = model(input)
        loss_test = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        test_losses.update(loss_test.data.item(), input.size(0))
        test_top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=test_losses,
                      top1=test_top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=test_top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', test_losses.avg, train_step)
        log_value('val_acc', test_top1.avg, train_step)

    model.train()
    return test_top1.avg


def train():
    global train_step, best_prec1
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        model.train()

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        train_step += 1
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        eval_interval = 100
        if i % args.print_freq == 0:
            print('Steps: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                train_step, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))

        if train_step % eval_interval == 0:
            prec1_test = validate(val_loader)

            # remember best prec@1 and save checkpoint
            is_best = prec1_test > best_prec1
            best_prec1 = max(prec1_test, best_prec1)
            save_checkpoint({
                'epoch': train_step + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
      
        device = loss.get_device()
        if train_step in step_list:
            feature_analysis(graphs, fc_features, model, criterion_summed, args.weight_decay, device,
                             args.num_classes, train_loader)

            feature_analysis(graphs_test, fc_features, model, criterion_summed, args.weight_decay, device, args.num_classes,
                             val_loader)

        with open('../figs/' + args.save_fig_dir + '/' + 'train.bin', "wb") as f:
            pickle.dump(graphs, f)
        with open('../figs/' + args.save_fig_dir + '/' + 'test.bin', "wb") as f:
            pickle.dump(graphs_test, f)

        if train_step == args.max_step:
            break
        # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss', losses.avg, train_step)
        log_value('train_acc', top1.avg, train_step)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

for epoch in itertools.count(start=1):
    #train(train_loader, val_loader, model, criterion, criterion_summed, optimizer, step_list)
    train()
    if train_step == args.max_step:
        break

print('Best accuracy: ', best_prec1)
ExecTime = time.perf_counter() - t
print('Running time: ', ExecTime)
