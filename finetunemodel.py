import argparse
import os
import shutil
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models 
from tensorboardX import SummaryWriter

NUM_CLASS = 2
CUDA_ENABLED = True

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch project argument parser')

parser.add_argument('--data_dir', type=str, default='./data',
                    help='Path to the celeba data directory.')

parser.add_argument('--config_path', type=str, default='./config.json',
                    help='Path to the config file')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0

writer = SummaryWriter()

conv_output_size = {"resnet18": 512,
                    "resnet34": 512,
                    "resnet50": 2048,
                    "resnet101": 2048,
                    "resnet152": 2048
                    }

class FineTuneResNet(nn.Module):
    def __init__(self, arch):
        super(FineTuneResNet, self).__init__()

        base_model = models.__dict__[arch](pretrained=True)

        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size[arch], NUM_CLASS)
        )
        self.modelName = 'resnet'

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

def main():
    global FLAGS, CONFIG, best_prec1
    FLAGS = parser.parse_args()
    CONFIG = _parse_config()

    model = FineTuneResNet(FLAGS.arch)

    model = torch.nn.DataParallel(model)
    if CUDA_ENABLED:
        model = _cudaize(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if CUDA_ENABLED:
        criterion = _cudaize(criterion)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                CONFIG["initial_lr"],
                                momentum=CONFIG["momentum"],
                                weight_decay=CONFIG["weight_decay"])

    # optionally resume from a checkpoint
    start_epoch = 0
    if FLAGS.resume:
        if os.path.isfile(FLAGS.resume):
            print("=> loading checkpoint '{}'".format(FLAGS.resume))
            checkpoint = torch.load(FLAGS.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(FLAGS.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(FLAGS.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(FLAGS.data_dir, 'training')
    validation_dir = os.path.join(FLAGS.data_dir, 'validation')
    test_dir = os.path.join(FLAGS.data_dir, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
            train_dir, 
            transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]
        ))

    validation_dataset = datasets.ImageFolder(
            validation_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]
        ))

    test_dataset = datasets.ImageFolder(
            test_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]
        ))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=FLAGS.workers, pin_memory=CUDA_ENABLED)
    
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=FLAGS.workers, pin_memory=CUDA_ENABLED)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=FLAGS.workers, pin_memory=CUDA_ENABLED)

    if FLAGS.evaluate:
        validate(validation_loader, model, criterion, 0)
        return

    for epoch in range(start_epoch, CONFIG["epochs"]):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(validation_loader, model, criterion, epoch*len(train_loader))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': FLAGS.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    
    print("The final accuracy of the model is")
    validate(test_loader, model, criterion, CONFIG["epochs"]*len(train_loader))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if CUDA_ENABLED:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write logs for tensorboard
        writer.add_scalar('training/learning_rate',  optimizer.param_groups[0]['lr'], global_step=epoch*len(train_loader)+i)
        writer.add_scalars('training/topN', {'top1': prec1[0]}, global_step=epoch*len(train_loader)+i)
        writer.add_scalar('training/loss', loss, global_step=epoch*len(train_loader)+i)

        if i % FLAGS.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, global_step):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        if CUDA_ENABLED:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # write logs for tensorboard
        writer.add_scalars('validation/topN', {'top1': prec1[0]}, global_step=global_step)
        writer.add_scalar('validation/loss', loss, global_step=global_step)

        if i % FLAGS.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate over time"""
    lr = CONFIG["initial_lr"] * (CONFIG["lr_decay_factor"] ** (epoch // CONFIG["num_epoch_per_decay"]))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def _parse_config():
    """parse training settings from a json file"""
    with open(FLAGS.config_path) as config_file:
        config_json = json.load(config_file)

    return config_json

def _cudaize(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

if __name__ == '__main__':
    main()