import argparse
import copy
import os
import stat
import sys
import time

import tabulate
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from models import sparse_variational_dropout as svdo, pre_resnet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy layerwise training of residual networks.')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', required=False, metavar='PATH',
                        help='path to checkpoints directory (default: ./checkpoints)')
    parser.add_argument('--data-dir', type=str, default='./cifar10', required=False, metavar='PATH',
                        help='path to dataset directory (default: ./cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, required=False, metavar='N',
                        help='input batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4, required=False, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--sparse', action='store_true', required=False,
                        help='train sparse network instead of dense one')
    parser.add_argument('--learning-rate', type=float, default=0.1, required=False, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, required=False, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, required=False, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--adam', action='store_true', required=False,
                        help='use ADAM optimizer instead of SGD')
    parser.add_argument('--save-freq', type=int, default=25, required=False, metavar='N',
                        help='number of epochs between checkpoints (default: 25)')
    parser.add_argument('--resume', type=str, default=None, required=False, metavar='CKPT',
                        help='resume training from checkpoint (default: None)')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help="device to train on (default: 'cuda:0' if torch.cuda.is_available() else 'cpu')")
    parser.add_argument('--num-epochs', type=int, default=200, required=False, metavar='N',
                        help='number of training epochs for every block (default: 200)')
    args = parser.parse_args()

    print('Preparing checkpoint folder: {}'.format(os.path.abspath(args.checkpoint_dir)))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'command.sh'), 'w') as f:
        f.write('#!/usr/bin/sh\n\npython {}\n'.format(' '.join(sys.argv)))
    st = os.stat(os.path.join(args.checkpoint_dir, 'command.sh'))
    os.chmod(os.path.join(args.checkpoint_dir, 'command.sh'), st.st_mode | stat.S_IEXEC)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print('Loading CIFAR-10 from {}'.format(os.path.abspath(args.data_dir)))
    dataloader = utils.get_cifar(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    print('Using {} ResNet110-v2'.format('sparse' if args.sparse else 'dense'))
    weak_classifiers = pre_resnet.make_resnet(sparse=args.sparse, sequential=True)
    model = pre_resnet.LayerwiseSequential()

    if args.sparse:
        criterion = svdo.SGVLB(model, len(dataloader['train'].dataset))
    else:
        criterion = nn.NLLLoss()

    start_epoch = 0
    start_block = 0
    if args.resume is not None:
        print('Resuming training from {}'.format(os.path.abspath(args.resume)))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_block = checkpoint['block']
        model.load_state_dict(checkpoint['model'])
        if args.adam:
            optimizer = optim.Adam(model.weak_classifier.parameters(),
                                   lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.weak_classifier.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        model.add_weak_classifier(weak_classifiers[0])
        if args.adam:
            optimizer = optim.Adam(model.weak_classifier.parameters(),
                                   lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.weak_classifier.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    columns = ['block', 'epoch', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    if args.sparse:
        columns = columns[:-1] + ['blk_sp', 'clf_sp', 'beta'] + columns[-1:]
        fmt = ['%s', '%s', '3.1e', '8.2f', '5.2f', '8.2f', '5.2f', '6.2f', '6.2f', '4.2f', '5.1f']
    else:
        fmt = ['%s', '%s', '3.1e', '8.6f', '5.2f', '8.6f', '5.2f', '5.1f']

    with open(os.path.join(args.checkpoint_dir, 'logs.csv'), 'w') as f:
        f.write('{}\n'.format(','.join(columns)))

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for block_idx in range(start_block, len(weak_classifiers)):
        if block_idx != start_block:
            model.to(torch.device('cpu'))
            old_weak_classifier = copy.copy(model.weak_classifier)
            model.add_weak_classifier(weak_classifiers[block_idx])
            model.residual_init(old_weak_classifier)
            if args.adam:
                optimizer = optim.Adam(model.weak_classifier.parameters(),
                                       lr=args.learning_rate, weight_decay=args.weight_decay)
            else:
                optimizer = optim.SGD(model.weak_classifier.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay, momentum=0.9)
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
            if args.sparse:
                criterion = svdo.SGVLB(model, len(dataloader['train'].dataset))
            else:
                criterion = nn.NLLLoss()
            start_epoch = 0

        block_checkpoint_dir = os.path.join(args.checkpoint_dir, 'block-{}'.format(block_idx + 1))
        os.makedirs(block_checkpoint_dir, exist_ok=True)
        for epoch in range(start_epoch, args.num_epochs):
            epoch_time = time.time()
            train_results = utils.train_model(model, optimizer, criterion, dataloader['train'], device=device)
            test_results = utils.validate_model(model, criterion, dataloader['test'], device=device)
            epoch_time = time.time() - epoch_time
            if (epoch - start_epoch + 1) % args.save_freq == 0 or epoch == args.num_epochs - 1:
                utils.make_checkpoint(
                    block_checkpoint_dir,
                    epoch + 1,
                    block=block_idx + 1,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict()
                )
            current_lr = args.learning_rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            values = [block_idx + 1, epoch + 1, train_results['loss'], train_results['accuracy'] * 100.0,
                      test_results['loss'], test_results['accuracy'] * 100.0, epoch_time]
            if args.sparse:
                block_compression = utils.compression(model.weak_classifier.block)
                classifier_compression = utils.compression(model.weak_classifier.classifier)
                values = values[:-1] + [block_compression, classifier_compression, criterion.beta] + values[-1:]
                if criterion.beta == 1.0:
                    lr_scheduler.step(test_results['loss'] / len(dataloader['train'].dataset))
                if epoch >= 49:
                    criterion.update_beta(step=0.02)
            else:
                lr_scheduler.step(test_results['loss'])
            table = tabulate.tabulate([values], columns, floatfmt=fmt)
            if (epoch - start_epoch) % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table)
            with open(os.path.join(os.path.abspath(args.checkpoint_dir), 'logs.csv'), 'a') as f:
                f.write('{}\n'.format(','.join([str(a) for a in values])))
