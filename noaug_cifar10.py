'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
# encoding=utf-8
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--lam', default = 0.1, type = float, help = 'lambda to adjust the angle distance')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
best_acc_lr_0 = 0  # when lr = 0.1 best test accuracy
best_acc_lr_1 = 0  # when lr = 0.01 best test accuracy
best_acc_lr_2 = 0  # when lr = 0.001 best test accuracy
best_acc_lr_3 = 0  # when lr = 0.0001 best test accuracy

# add--------------------
global_writer = 0
global_num_classes = 0
global_record = 0

def isequal(a, b):
    if abs(a-b) < 0.0000001:
        return True
    return False

def main():
    global best_acc
    global best_acc_lr_0
    global best_acc_lr_1
    global best_acc_lr_2
    global best_acc_lr_3
    global global_writer
    global global_num_classes
    global global_record
    
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        global_num_classes = 10  # add
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
        global_num_classes = 100  # add


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model   
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )        
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-%d-' % (global_num_classes) + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        if os.path.exists(os.path.join(args.checkpoint, 'log.txt')):
            os.system('rm ' + os.path.join(args.checkpoint, 'log.txt'))
            print('exist log.txt and rm log.txt')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc.'])
        
        if os.path.exists(os.path.join(args.checkpoint, 'info.txt')):
            os.system('rm ' + os.path.join(args.checkpoint, 'info.txt'))
            print('exist info.txt and rm info.txt')
        global_writer = open(os.path.join(args.checkpoint, 'info.txt'), 'a', 0)
        print('random seed = %d' % args.manualSeed, file = global_writer)
        
        if os.path.exists(os.path.join(args.checkpoint, 'record.txt')):
            os.system('rm ' + os.path.join(args.checkpoint, 'record.txt'))
            print('exist record.txt and rm record.txt')
        global_record = open(os.path.join(args.checkpoint, 'record.txt'), 'a', 0)
        print('random seed = %d' % args.manualSeed, file = global_record)


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, angleW, theta = train(trainloader, model, criterion, optimizer, epoch, use_cuda, args.lam)
        test_loss, test_acc, confusion_matrix, total_wrong = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, best_acc])
        
        print('epoch: {}'.format(epoch), file = global_record)
        print('test_acc: {}'.format(test_acc), file = global_record)
        print('angleW:', file = global_record)
        print('{} {}'.format(angleW.size(0), angleW.size(1)), file = global_record)
        for val in angleW:
            for w in val:
                global_record.write('{} '.format(w))
            global_record.write('\n')
        
        print('theta:', file = global_record)
        print('{} {}'.format(theta.size(0), theta.size(1)), file = global_record)
        for val in theta:
            for w in val:
                global_record.write('{} '.format(w))
            global_record.write('\n')
        
        print('confusion_matrix:', file = global_record)
        print('{} {}'.format(confusion_matrix.size(0), confusion_matrix.size(1)), file = global_record)
        for val in confusion_matrix:
            for w in val:
                global_record.write('{} '.format(int(w)))
            global_record.write('\n')
        
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        print('best_acc: {}'.format(best_acc), file = global_record)
        print('-'*100, file = global_record)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, 
            is_best,
            checkpoint=args.checkpoint,
            filename='checkpoint.pth.tar',
            bestfilename='model_best.pth.tar')
        
        print(state['lr'])
        if isequal(state['lr'], 0.1):
            print('lr==0.1')
            is_best_lr_0 = test_acc > best_acc_lr_0
            best_acc_lr_0 = max(test_acc, best_acc_lr_0)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc_lr_0,
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best_lr_0,
                checkpoint=args.checkpoint,
                filename='checkpoint_lr_0.pth.tar',
                bestfilename='model_best_lr_0.pth.tar')
        if isequal(state['lr'], 0.01):
            print('lr==0.01')
            is_best_lr_1 = test_acc > best_acc_lr_1
            best_acc_lr_1 = max(test_acc, best_acc_lr_1)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc_lr_1,
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best_lr_1,
                checkpoint=args.checkpoint,
                filename='checkpoint_lr_1.pth.tar',
                bestfilename='model_best_lr_1.pth.tar')
        if isequal(state['lr'], 0.001):
            print('lr==0.001')
            is_best_lr_2 = test_acc > best_acc_lr_2
            best_acc_lr_2 = max(test_acc, best_acc_lr_2)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc_lr_2,
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best_lr_2,
                checkpoint=args.checkpoint,
                filename='checkpoint_lr_2.pth.tar',
                bestfilename='model_best_lr_2.pth.tar')
        if isequal(state['lr'], 0.0001):
            print('lr==0.0001')
            is_best_lr_3 = test_acc > best_acc_lr_3
            best_acc_lr_3 = max(test_acc, best_acc_lr_3)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc_lr_3,
                    'optimizer' : optimizer.state_dict(),
                }, 
                is_best_lr_3,
                checkpoint=args.checkpoint,
                filename='checkpoint_lr_3.pth.tar',
                bestfilename='model_best_lr_3.pth.tar')

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    # add--------------------
    print ('Best acc:\n lr=0.1 acc={:.4f}\n lr=0.01 acc={:.4f}\n lr=0.001 acc={:.4f}\n'.format(best_acc_lr_0, best_acc_lr_1, best_acc_lr_2))
    print('Best acc: {:.4f}\n'.format(best_acc))
    
    print('Best acc:', file = global_writer)
    print(best_acc, file = global_writer)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda, lam):
    global global_writer
    global global_num_classes
    
    # switch to train mode
    model.train()
    
    print('lam = ', lam)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    
    # add---------------------
    fixed   = torch.ones(global_num_classes, 1)  # [class, 1]
    if use_cuda:
        fixed = fixed.cuda()
    fixed = torch.autograd.Variable(fixed, requires_grad = False)
    
    # add--------------------
    theta_clone = 0
    angleW_clone = 0
    
    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)  # batch_size x n_class
        
        # add--------------------
        angleW  = model.module.fc.weight_v
        
        norm = angleW.data.norm(2, 1, keepdim=True).expand(angleW.data.size())
        angleW.data.div_(norm)
        # print(angleW.norm(dim=1))
        # print(angleW)
        # exit()
        
        angleWt = angleW.t() # 256 x n_class
        W       = angleWt.mm(fixed)  # fixed: n_class x 1    W: 256 x 1
        Wt      = W.t()  # 1 x 256
        cos     = Wt.mm(W)  # 1 x 1
        cos     = cos * (1.0 * lam / global_num_classes / global_num_classes)
        cos_loss_part = cos.data[0][0]
        
        loss = criterion(outputs, targets) + cos

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0][0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
        if batch_idx == len(trainloader) - 1:
        # if batch_idx > -1:
            x = angleW.data     # n_class x 256
            angleW_clone = x.clone()
            y = x.mm(x.t())     # n_class x n_class
            n = x.norm(2, 1, keepdim=True)    # p==2, dim==1  n_class x 1   sqrt(sigma(xi^xi)(i=1->n)) length
            m = n.mm(n.t())     # n_class x n_class
            c = y / m           # normlize
            c[c > 1.0] = 1.0
            import math
            PI = math.acos(-1.0)
            theta      = c.acos()
            theta      = theta / PI
            theta      = theta * 180.0
            print('batch_idx = ' + str(batch_idx), file = global_writer)
            # print(theta)
            theta_clone = theta.clone()
            print(theta, file = global_writer)
            
            print('min = ' + str(theta[theta > 10.0].min()), file = global_writer)
            print('max = ' + str(theta.max()), file = global_writer)
            print('mean = ' + str(theta[theta > 10.0].mean()), file = global_writer)
        
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # add--------------------
        # TODO renormalize the angleW
        # norm = angleW.data.norm(2, 1).expand_as(angleW.data)
        # angleW.data.div_(norm)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # add--------------------
        # w = model.module.fc.weight.data
        # w = w.norm(2, 1).squeeze()
        # mw = w.mean()
        #s = ''
        #for x in range(w.size(0)):
        #    s += '%6.3f ' % w[x]
        #print(s)
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
                    batch   = batch_idx + 1,
                    size    = len(trainloader),
                    data    = data_time.avg,
                    bt      = batch_time.avg,
                    total   = bar.elapsed_td,
                    eta     = bar.eta_td,
                    loss    = losses.avg,
                    top1    = top1.avg,
                    top5    = top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, angleW_clone, theta_clone)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc
    global global_writer
    global global_num_classes

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    
    # add--------------------
    total_wrong = 0
    confusion_matrix = torch.zeros(global_num_classes, global_num_classes)
    
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        # add--------------------
        _, predicted = torch.max(outputs.data, 1)  # val, index
        for i in xrange(targets.data.size(0)):
            p = predicted[i]
            v = targets.data[i]
            if p != v:
                confusion_matrix[v][p] += 1
                total_wrong += 1
            else:
                confusion_matrix[v][p] += 1
        
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    
    # print(confusion_matrix)
    # print('total_wrong = ', total_wrong)

    print('test acc = ', top1.avg, file = global_writer)
    return (losses.avg, top1.avg, confusion_matrix, total_wrong)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', bestfilename='model_best.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, bestfilename))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()