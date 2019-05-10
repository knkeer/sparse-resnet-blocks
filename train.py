import argparse
import os
import stat
import sys
import time

import tabulate
import torch
import torch.nn as nn
import torch.optim as optim

import utils
from models import pre_resnet
from models import sparse_variational_dropout as svdo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end training of residual networks.')
    parser.add_argument('--dir', type=str, default='./checkpoints', required=False,
                        help='path to the checkpoints directory (default: ./checkpoints)')
    parser.add_argument('--data-path', type=str, default='./cifar10', required=False, metavar='PATH',
                        help='path to the dataset directory (default: ./cifar10)')
    parser.add_argument('--batch-size', type=int, default=64, required=False, metavar='N',
                        help='input batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4, required=False, metavar='N',
                        help='number of workers used for training (default: 4)')
    parser.add_argument('--sparse', action='store_true',
                        help='train sparse model instead of dense one (default: False)')
    parser.add_argument('--num-epochs', type=int, default=150, required=False, metavar='N',
                        help='number of epochs to train (default: 150)')
    parser.add_argument('--learning-rate', type=float, default=0.1, required=False, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, required=False, metavar='WD',
                        help='weight decay of parameters (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=1, required=False, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--adam', action='store_true',
                        help='use ADAM optimizer instead of SGD (default: False)')
    parser.add_argument('--save-freq', type=int, default=25, required=False, metavar='N',
                        help='save frequency (default: 25)')
    parser.add_argument('--eval-freq', type=int, default=5, required=False, metavar='N',
                        help='evaluation frequency (default: 5)')
    parser.add_argument('--resume', type=str, default=None, required=False, metavar='CKPT',
                        help='resume training from checkpoint (default: None)')
    args = parser.parse_args()

    print('Preparing checkpoints folder: {}'.format(os.path.abspath(args.dir)))
    # create dir if it doesn't exist
    os.makedirs(os.path.abspath(args.dir), exist_ok=True)
    # save executable script for reproducing
    with open(os.path.join(os.path.abspath(args.dir), 'command.sh'), 'w') as f:
        f.write('#!/usr/bin/env python3\n\n')
        f.write('{}\n'.format(' '.join(sys.argv)))
    st = os.stat(os.path.join(os.path.abspath(args.dir), 'command.sh'))
    os.chmod(os.path.join(os.path.abspath(args.dir), 'command.sh'), st.st_mode | stat.S_IEXEC)

    # fix seeds and enable CuDNN
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load dataset and model
    print('Loading CIFAR-10 from {}'.format(os.path.abspath(args.data_path)))
    cifar = utils.get_cifar(args.data_path, batch_size=args.batch_size, num_workers=args.num_workers)
    print('Using {} ResNet110-v2'.format('sparse' if args.sparse else 'dense'))
    model = pre_resnet.make_resnet(depth=110, sparse=args.sparse, sequential=False)
    # set criterion and optimizer
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                              weight_decay=args.weight_decay, momentum=0.9)
    if args.sparse:
        criterion = svdo.SGVLB(model, len(cifar['train'].dataset))
    else:
        criterion = nn.NLLLoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    start_epoch = 0
    if args.resume is not None:
        print('Resuming training from {}'.format(os.path.abspath(args.resume)))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['checkpoint'])

    columns = ['epoch', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    fmt = {'lr': '3.1e', 'tr_acc': '5.2f', 'te_acc': '5.2f', 'time': '5.1f'}
    if args.sparse:
        columns = columns[:-1] + ['sp', 'beta'] + columns[-1:]
        fmt = ['%s', '3.1e', '8.2f', '5.2f', '8.2f', '5.2f', '6.2f', '4.2f', '5.1f']
    else:
        fmt = ['%s', '3.1e', '8.6f', '5.2f', '8.6f', '5.2f', '5.1f']
    with open(os.path.join(os.path.abspath(args.dir), 'logs.csv'), 'w') as f:
        f.write('{}\n'.format(','.join(columns)))

    for epoch in range(start_epoch, args.num_epochs):
        time_epoch = time.time()
        train_result = utils.train_model(model, optimizer, criterion, cifar['train'])
        test_result = utils.validate_model(model, criterion, cifar['test'])
        time_epoch = time.time() - time_epoch
        if (epoch != start_epoch and (epoch - start_epoch) % args.save_freq == 0) or epoch == args.num_epochs - 1:
            utils.make_checkpoint(
                args.dir, epoch + 1, model=model.state_dict(),
                optimizer=optimizer.state_dict()
            )
        current_lr = args.learning_rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        values = [epoch + 1, current_lr, train_result['loss'], train_result['accuracy'] * 100.0,
                  test_result['loss'], test_result['accuracy'] * 100.0, time_epoch]
        if args.sparse:
            values = values[:-1] + [utils.compression(model), criterion.beta] + values[-1:]
            criterion.update_beta()
            if criterion.beta == 1.0:
                lr_scheduler.step(test_result['loss'] / len(cifar['train'].dataset))
        else:
            lr_scheduler.step(test_result['loss'])
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt=fmt)
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        with open(os.path.join(os.path.abspath(args.dir), 'logs.csv'), 'a') as f:
            f.write('{}\n'.format(','.join([str(a) for a in values])))
