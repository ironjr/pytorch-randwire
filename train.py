import argparse
import os
import time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from randwire import RandWireSmall78, RandWireRegular109, RandWireRegular154
from loss import CEWithLabelSmoothingLoss
from scheduler import CosineAnnealingWithRestartsLR
from util import *
from logger import Logger


def main(args):
    iteration = 0

    # Tensorboard loggers
    log_root = './log'
    log_names = ['', 'train_loss', 'test_loss']
    log_dirs = list(map(lambda x: os.path.join(log_root, args.label, x), log_names))
    if not os.path.isdir(log_root):
        os.mkdir(log_root)
    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
    _, train_logger, val_logger = \
            list(map(lambda x: Logger(x), log_dirs))

    # Checkpoint save directory
    checkpoint_root = './checkpoint'
    if not os.path.isdir(checkpoint_root):
        os.mkdir(checkpoint_root)
    checkpoint_dir = os.path.join(checkpoint_root, args.label)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # Data transform
    print('==> Preparing data ..')
    traindir = os.path.join(args.data_root, 'train')
    valdir = os.path.join(args.data_root, 'val')
    traintf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]),
    ])
    valtf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]),
    ])
    trainset = datasets.ImageFolder(traindir, traintf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valset = datasets.ImageFolder(valdir, valtf)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model and optimizer
    start_iter = 0
    if args.resume:
        # Load from preexisting models
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        # Model from existing random graphs
        graphs = checkpoint['graphs']
        if args.model in ['small78', 'small']:
            model, _ = RandWireSmall78(Gs=graphs, Gopt=True)
        elif args.model in ['regular109', 'medium']:
            model, _ = RandWireRegular109(Gs=graphs, Gopt=True)
        elif args.model in ['regular154', 'large']:
            model, _ = RandWireRegular154(Gs=graphs, Gopt=True)
        else:
            raise NotImplementedError
        model.load_state_dict(checkpoint['model'])

        optimizer = optim.SGD(
                group_weight(model),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer.load_state_dict(checkpoint['optim'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        # Overwrite default hyperparameters for new run
        group_decay, group_no_decay = optimizer.param_groups
        group_decay['lr'] = args.lr
        group_decay['momentum'] = args.momentum
        group_decay['weight_decay'] = args.weight_decay
        group_no_decay['lr'] = args.lr
        group_no_decay['momentum'] = args.momentum

        args.start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iteration']
        print("Last loss: %.3f" % (checkpoint['loss']))
        print("Training start from epoch %d iteration %d" % (args.start_epoch, start_iter))
    else:
        # Start from scratch
        print('==> Generating new model..')

        # TODO put this configuration out from the main method
        graph_type = 'WS'
        graph_params = {
            'P': 0.75,
            'K': 4,
        }
        if args.model in ['small78', 'small']:
            model, graphs = RandWireSmall78(
                    model=graph_type,
                    params=graph_params,
                    seeds=None)
        elif args.model in ['regular109', 'medium']:
            model, graphs = RandWireRegular109(
                    model=graph_type,
                    params=graph_params,
                    seeds=None)
        elif args.model in ['regular154', 'large']:
            model, graphs = RandWireRegular154(
                    model=graph_type,
                    params=graph_params,
                    seeds=None)
        else:
            raise NotImplementedError

        optimizer = optim.SGD(
                group_weight(model),
                lr=args.lr,
                momentum=0.9,
                weight_decay=5e-5)

        print('Model generated with : ', graph_type, 'method with params', graph_params, 'and seed', None)

    # Use cosine annealing scheduler unless noted
    scheduler = None
    if not args.no_cosine_annealing:
        scheduler = CosineAnnealingWithRestartsLR(
                optimizer,
                T_max=args.t_max,
                T_mult=args.t_mult)

    # Criterion has no internal parameters
    criterion = CEWithLabelSmoothingLoss

    # Use CUDA and enable multi-GPU learning
    model = torch.nn.DataParallel(model,
            device_ids=range(torch.cuda.device_count()))
    model.cuda()
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Main run
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        if scheduler is not None:
            scheduler.step()
            if scheduler.save_flag:
                save('restart' + str(epoch - 1), model, graphs, optimizer, epoch)
        train(trainloader, model, graphs, criterion, optimizer, epoch,
                train_logger=train_logger, save_every=args.save_every,
                start_iter=start_iter)
        test(valloader, model, graphs, criterion, epoch,
                (epoch + 1) * len(trainloader), optimizer=optimizer,
                val_logger=val_logger)


# Train
def train(trainloader, model, graphs, criterion, optimizer, epoch, start_iter=0,
        train_logger=None, save_every=1000):
    print('\nEpoch: %d' % epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #  top5 = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
        # Data loading time
        data_time.update(time.time() - end)

        # Offset correction
        idx = batch_idx + start_iter

        # FORWARD
        input_vars = Variable(inputs.cuda())
        target_vars = Variable(targets.cuda())

        outputs = model(input_vars)
        loss = criterion(outputs, target_vars, eps=0.1)

        # Update log
        prec = accuracy(outputs.data, target_vars, topk=(1,))
        top1.update(prec[0], inputs.size(0))
        losses.update(loss.data, inputs.size(0))

        # BACKWARD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Batch evaluation time
        batch_time.update(time.time() - end)

        # Log current performance
        step = idx + len(trainloader) * epoch
        if train_logger is not None:
            info = {
                'loss': loss.data,
                'average loss': losses.avg,
                'top1 precision': np.asscalar(top1.val.cpu().numpy()),
                'top1 average precision': np.asscalar(top1.avg.cpu().numpy()),
            }
            for tag, value in info.items():
                train_logger.scalar_summary(tag, value, step + 1)

        tqdm.write('batch (%d/%d) | loss: %.3f | avg_loss: %.3f | Prec@1: %.3f %% (%.3f %%)'
                % (batch_idx, len(trainloader), losses.val, losses.avg,
                    np.asscalar(top1.val.cpu().numpy()),
                    np.asscalar(top1.avg.cpu().numpy())))

        # Save at every specified cycles
        if (idx + 1) % save_every == 0:
            save('train' + str(step), model, graphs, optimizer, losses.avg,
                    epoch, idx + 1)

        # Update the base time
        end = time.time()

        # Finish when total iterations match the number of batches
        if start_iter != 0 and (idx + 1) % len(trainloader) == 0:
            break

# Test
def test(valloader, model, graphs, criterion, epoch, iteration, val_logger=None, optimizer=None):
    print('\nTest')

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #  top5 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(valloader)):
            # FORWARD
            input_vars = Variable(inputs.cuda())
            target_vars = Variable(targets.cuda())

            outputs = model(input_vars)
            loss = criterion(outputs, target_vars, eps=0.1)

            # Batch evaluation time
            batch_time.update(time.time() - end)

            # Update log
            prec = accuracy(outputs.data, target_vars, topk=(1,))
            top1.update(prec[0], inputs.size(0))
            losses.update(loss.data, inputs.size(0))

            tqdm.write('batch (%d/%d) | loss: %.3f | avg_loss: %.3f | Prec@1: %.3f %% (%.3f %%)' 
                    % (batch_idx, len(valloader), losses.val, losses.avg,
                        np.asscalar(top1.val.cpu().numpy()),
                        np.asscalar(top1.avg.cpu().numpy())))

            # Update the base time
            end = time.time()

    # Save and log
    save('test' + str(epoch), model, graphs, optimizer, losses.avg, epoch + 1)

    if val_logger is not None:
        info = {
            'average loss': losses.avg,
            'top1 average precision': np.asscalar(top1.avg.cpu().numpy()),
        }
        for tag, value in info.items():
            val_logger.scalar_summary(tag, value, iteration)
    print('average validation loss: %.3f' % (losses.avg))
    print('average top1 precision : %.3f' % (np.asscalar(top1.avg.cpu().numpy())))


# Save checkpoints
def save(label, model, graphs, optimizer, loss=float('inf'), epoch=0, iteration=0):
    tqdm.write('==> Saving checkpoint')
    state = {
        'model': model.module.state_dict(),
        'graphs': graphs,
        'optim': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'iteration': iteration,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.label + '/ckpt_' + label + '.pth')
    tqdm.write('==> Save done!')


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'CUDA is required!'

    parser = argparse.ArgumentParser(description='PyTorch RandWireNet Training')
    parser.add_argument('--label', default='default', type=str,
            help='labels checkpoints and logs saved under')
    parser.add_argument('--model', default='regular109', type=str,
            choices=('small78', 'small', 'regular109', 'medium', 'regular154', 'large'),
            help='backbone network to use in fpn')
    parser.add_argument('--num-workers', default=2, type=int,
            help='number of workers in dataloader')
    parser.add_argument('--data-root', default='../common/datasets/ImageNet', type=str,
            help='ImageNet12 folder where train/ and val/ belong')
    parser.add_argument('--no-cosine-annealing', action='store_true',
            help='use uniform scheduling instead of cosine annealing')
    parser.add_argument('--t-max', default=10, type=int,
            help='restart period of cosine annealing')
    parser.add_argument('--t-mult', default=1.0, type=float,
            help='factor of increment of restart period each restart')
    parser.add_argument('--lr', default=1e-2, type=float,
            help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
            help='SGD momentum')
    parser.add_argument('--weight-decay', default=5e-5, type=float,
            help='weight decay for SGD (small: 5e-5, large: 1e-5)')
    parser.add_argument('--batch-size', default=128, type=int,
            help='size of a minibatch')
    parser.add_argument('--start-epoch', default=0, type=int,
            help='epoch index to start log')
    parser.add_argument('--num-epochs', default=1, type=int,
            help='number of epochs to run')
    parser.add_argument('--save-every', default=2000, type=int,
            help='save cycle during train mode')
    parser.add_argument('--resume', '-r', action='store_true',
            help='resume from checkpoint')
    parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth', type=str,
            help='path to the checkpoint to load')
    args = parser.parse_args()

    # Run main routine
    main(args)
