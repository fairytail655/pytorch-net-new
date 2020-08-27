import argparse
import os
import time
import torch
import torch.optim
import torch.utils.data
import models
import math
import json
import sys
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from vl_draw import *
from data import get_dataset
from utils import *
from datetime import datetime
from torchsummary import summary
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='resnet20_my',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet20_my',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: lenet)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--device', default='gpu', type=str, metavar='OPT',
                    help='choose to use CPU or GPU')

def main():
    global my_logging
    global args, best_prec
    global progress, task2, task3
    global input_size, in_dim
    global thread_train, thread_val
    global thread_hm
    global device
    global epochs

    best_prec = 0
    set_value_list = [
        'vgg_selfbinaring',
        'vgg_my',
        'resnet20_my',
        'resnet20_my_1w1a',
    ]

    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = './tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    my_logging = MyLogging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv')

    my_logging.info("saving to {}".format(save_path))
    my_logging.info("run arguments: {}".format(args))

    if args.device.upper() == 'GPU': 
        device = torch.device('cuda')
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    my_logging.info("using device: {}".format(args.device.upper()))

    # create model
    model = models.__dict__[args.model]
    model_config = {'dataset': args.dataset}
    model = model(**model_config)
    my_logging.info("created model \"{}\" with configuration: {}"
                    .format(args.model, model_config))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['model_state_dict'])
        my_logging.info("loaded checkpoint '{}' (epoch {})"
                        .format( args.evaluate, checkpoint['epoch']))
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_file):
            my_logging.info("loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            my_logging.info("loaded checkpoint '{}' (epoch {})"
                            .format(checkpoint_file, checkpoint['epoch']))
        else:
            parser.error('no checkpoint found: {}'.format(checkpoint_file))

    if args.dataset == 'mnist':
        input_size = 28
        in_dim = 1
    elif args.dataset == 'cifar10':
        input_size = 32
        in_dim = 3

    train_config = getattr(model, 'train_config', {})
    if args.dataset in train_config:
        train_config = train_config[args.dataset]
    else:
        train_config = {}

    if 'epochs' in train_config:
        epochs = train_config['epochs']
    else:
        epochs = args.epochs

    if 'batch_size' in train_config:
        batch_size = train_config['batch_size']
    else:
        batch_size = args.batch_size

    if 'opt_config' in train_config:
        opt_config = train_config['opt_config']
    else:
        opt_config = { 0: {
                            'optimizer': args.optimizer,
                            'lr': args.lr,
                            'momentum': args.momentum,
                            'weight_decay': args.weight_decay
                        }
        }

    if 'transform' in train_config:
        transform = train_config['transform']
    else:
        transform = {'train': None, 'eval': None}

    my_logging.info("\n----------------------------------------------\n"
                    "epochs: {}\tbatch_size: {}\n"
                    "opt_config: {}\n"
                    "transform: {}\n"
                    "----------------------------------------------"
                    .format(epochs, batch_size, opt_config, transform)
    )

    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    # print net struct
    if args.model in set_value_list:
        model.set_value(0, 10, False)
    if args.dataset == 'mnist':
        summary(model, (1, 28, 28))
    elif args.dataset == 'cifar10':
        summary(model, (3, 32, 32))

    val_data = get_dataset(args.dataset, 'eval', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    my_logging.info('val dataset size: {}'.format(len(val_data)))

    if args.evaluate:
        with Progress("[progress.description]{task.description}{task.completed}/{task.total}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeRemainingColumn(),
                    auto_refresh=False) as progress:
            task3 = progress.add_task("[yellow]validating:", total=math.ceil(len(val_data)/args.batch_size))
            if args.model in set_value_list:
                model.set_value(0, 10, False)
            val_loss, val_prec = validate(val_loader, model, criterion, 0)
            my_logging.info('Evaluate {0}\t'
                        'Validation Loss {val_loss:.4f} \t'
                        'Validation Prec@1 {val_prec1:.3f} \t'
                        .format(args.evaluate, val_loss=val_loss, val_prec1=val_prec))
        return

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    my_logging.info('train dataset size: {}'.format(len(train_data)))

    # visualDL scalar init
    # command: visualdl --logdir ./vl_log --port 8080
    scalar_path = os.path.join("./vl_log/scalar", args.model)
    my_logging.info('save visualDL scalar log in: {}'.format(scalar_path))
    thread_train = DrawScalar(os.path.join(scalar_path, "train"), args.model)
    thread_val = DrawScalar(os.path.join(scalar_path, "val"), args.model)
    thread_train.start()
    thread_val.start()

    # visualDL histogram init
    hm_layer_names = [
                        'conv1.weight',
                        # 'conv6.weight',
                        'layer3.2.conv2.weight',
                        'fc.weight',
                    ]
    histogram_path = os.path.join("./vl_log/histogram", args.model)
    my_logging.info('save visualDL histogram log in: {}'.format(histogram_path))
    thread_hm = {}
    for name in hm_layer_names:
        thread_hm[name] = DrawHistogram(histogram_path, name)
        thread_hm[name].start()

    # first draw visualDL histogram
    for name in model.state_dict():
        if name in hm_layer_names:
            value = model.state_dict()[name].cpu().numpy().reshape(-1)
            thread_hm[name].set_value(0, value)

    # print progressor
    with Progress("[progress.description]{task.description}{task.completed}/{task.total}",
                  BarColumn(),
                  "[progress.percentage]{task.percentage:>3.0f}%",
                  TimeRemainingColumn(),
                  auto_refresh=False) as progress:
        task1 = progress.add_task("[red]epoch:", total=epochs)
        task2 = progress.add_task("[blue]training:", total=math.ceil(len(train_data)/batch_size))
        task3 = progress.add_task("[yellow]validating:", total=math.ceil(len(val_data)/batch_size))

        for i in range(args.start_epoch):
            progress.update(task1, advance=1, refresh=True)

        begin = time.time()
        for epoch in range(args.start_epoch, epochs):
            start = time.time()
            optimizer = adjust_optimizer(optimizer, epoch, opt_config)

            # update param 'v' in SelfBinarize
            if args.model in set_value_list:
                model.set_value(epoch, epochs, True)

            # train for one epoch
            train_loss, train_prec = train(
                train_loader, model, criterion, epoch, optimizer)

            # evaluate on validation set
            val_loss, val_prec = validate(
                val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = val_prec > best_prec
            best_prec = max(val_prec, best_prec)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'model_state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'best_prec': best_prec,
            }, is_best, path=save_path)
            my_logging.debug('\n----------------------------------------------\n'
                        'Epoch: [{0}/{1}] Cost_Time: {2:.2f}s\t'
                        'current_lr: {3:.7f}\t'
                        'Training Loss {train_loss:.4f} \t'
                        'Training Prec {train_prec:.3f} \t'
                        'Validation Loss {val_loss:.4f} \t'
                        'Validation Prec {val_prec:.3f} \n'
                        '----------------------------------------------'
                        .format(epoch + 1, epochs, time.time()-start, optimizer.state_dict()['param_groups'][0]['lr'],
                                train_loss=train_loss, val_loss=val_loss, 
                                train_prec=train_prec, val_prec=val_prec))

            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                        train_prec=train_prec, val_prec=val_prec)
            results.save()

            # draw visualDL scalar
            thread_train.set_value(epoch, {'loss': train_loss, 'acc': train_prec})
            thread_val.set_value(epoch, {'loss': val_loss, 'acc': val_prec})

            #draw visualDL histogram
            for name in model.state_dict():
                if name in hm_layer_names:
                    value = model.state_dict()[name].cpu().numpy().reshape(-1)
                    thread_hm[name].set_value(epoch+1, value)

            # update epoch progressor
            progress.update(task1, advance=1, refresh=True)

    my_logging.info('\n=================================================================\n'
                'Whole Cost Time: {0:.2f}min      Best Validation Prec {1:.3f}\n'
                '================================================================='.format((time.time()-begin)/60, best_prec))
    
def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    # data_time = AverageMeter()
    losses = AverageMeter()
    precisions = AverageMeter()

    start = time.time()

    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        # data_time.update(time.time() - start)
        target = target.to(device)
        inputs = inputs.to(device)

        if not training:
            with torch.no_grad():
                inputs = Variable(inputs)
                target = Variable(target)
                # compute output
                output = model(inputs)
        else:
            inputs = Variable(inputs)
            target = Variable(target)
            # compute output
            output = model(inputs)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        precisions.update(prec.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - start)

        # update train and val progressor
        if training:
            progress.update(task2, advance=1, refresh=True)
        else:
            progress.update(task3, advance=1, refresh=True)

        # if i % args.print_freq == 0:
        #     logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
        #                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
        #                      epoch, i, len(data_loader),
        #                      phase='TRAINING' if training else 'EVALUATING',
        #                      batch_time=batch_time,
        #                      data_time=data_time, loss=losses, top1=top1))

    if not training:
        progress.update(task3, completed=0)
    else:
        progress.update(task2, completed=0)

    return losses.avg, precisions.avg

def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)

def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Main KeyboardInterrupt....")
    try:
        thread_train.stop()
        thread_train.join()
        thread_val.stop()
        thread_val.join()
        for thread in thread_hm.keys():
            thread_hm[thread].stop()
            thread_hm[thread].join()
    except:
        pass
    sys.exit(1)
