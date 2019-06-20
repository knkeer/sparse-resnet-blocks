import argparse
import os
import time

import tabulate
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training import utils
from models.glt_models import LinearClassifier


torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end training of residual networks.')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints-e2e', required=False, metavar='PATH',
                        help='path to checkpoints directory (default: ./checkpoints-e2e)')
    parser.add_argument('--data-dir', type=str, default='./cifar10', required=False, metavar='PATH',
                        help='path to dataset directory (default: ./cifar10)')
    parser.add_argument('--batch-size', type=int, default=128, required=False, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=6, required=False, metavar='N',
                        help='number of workers (default: 6)')
    parser.add_argument('--sparse', action='store_true', required=False,
                        help='train sparse network instead of dense one')
    parser.add_argument('--learning-rate', type=float, default=0.1, required=False, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, required=False, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=12345, required=False, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--adam', action='store_true', required=False,
                        help='use ADAM optimizer instead of SGD')
    parser.add_argument('--save-freq', type=int, default=25, required=False, metavar='N',
                        help='number of epochs between checkpoints (default: 25)')
    parser.add_argument('--resume', type=str, default=None, required=False, metavar='CKPT',
                        help='resume training from checkpoint (default: None)')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help="device to train on (default: 'cuda:0' if torch.cuda.is_available() else 'cpu')")
    parser.add_argument('--num-epochs', type=int, default=150, required=False, metavar='N',
                        help='number of training epochs for every block (default: 150)')
    parser.add_argument('--depth', type=int, default=110, required=False, metavar='N',
                        help='depth of residual network (default: 110)')
    parser.add_argument('--reverse', action='store_true', required=False,
                        help='reverse layer order of residual network (default: off)')
    parser.add_argument('--width', type=int, default=1, required=False,
                        help='width of residual network (default: 1)')
    args = parser.parse_args()

    utils.set_random_seed(args.seed)

    print('Preparing checkpoint folder: {}'.format(os.path.abspath(args.checkpoint_dir)))
    utils.prepare_checkpoint_folder(args)

    print('Loading CIFAR-10 from {}'.format(os.path.abspath(args.data_dir)))
    dataloader = utils.get_dataloader(args)

    print('Using {} ResNet{}-v2'.format('sparse' if args.sparse else 'dense', args.depth))
    layers = utils.get_resnet_layers(args)
    linear_classifier = LinearClassifier(layers[-1].out_channels, sparse=args.sparse)
    model = nn.Sequential(*layers, linear_classifier)

    print('Using {} optimizer with initial learning rate {} and weight decay {}'.format(
        'ADAM' if args.adam else 'SGD', args.learning_rate, args.weight_decay
    ))
    optimizer = utils.get_optimizer(model, args)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5)

    # set starting positions and load checkpoint if necessary
    start_epoch = 0
    if args.resume is not None:
        if not os.path.isfile(args.resume):
            raise ValueError('Resume file {} not found'.format(os.path.abspath(args.resume)))
        print('Resuming training from {}'.format(os.path.abspath(args.resume)))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # set loss function
    criterion = nn.NLLLoss()

    # prepare logs
    columns, fmt = utils.get_columns(args, end_to_end=True)
    with open(os.path.join(args.checkpoint_dir, 'logs.csv'), 'w') as f:
        f.write('{}\n'.format(','.join(columns)))

    # get device to train on
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(start_epoch, args.num_epochs):
        # get beta
        beta = utils.get_beta(epoch) if args.sparse else None

        # train and evaluate model
        epoch_time = time.time()
        train_results = utils.train_model(model, optimizer, criterion, dataloader['train'], device=device, beta=beta)
        test_results = utils.evaluate_model(model, criterion, dataloader['test'], device=device, beta=beta)
        epoch_time = time.time() - epoch_time

        # save model if necessary
        if (epoch - start_epoch + 1) % args.save_freq == 0 or epoch == args.num_epochs - 1:
            utils.make_checkpoint(
                args.checkpoint_dir,
                epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                lr_scheduler=lr_scheduler.state_dict()
            )

        # get current learning rate
        current_lr = args.learning_rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        # get data about this training epoch
        values = [epoch + 1, current_lr, train_results['loss'], train_results['accuracy'],
                  test_results['loss'], test_results['accuracy'], epoch_time]
        if args.sparse:
            values = values[:-1] + [test_results['kl'], utils.compression_ratio(model), beta] + values[-1:]

        # print data to standard output
        table = tabulate.tabulate([values], columns, floatfmt=fmt)
        if (epoch - start_epoch) % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        # save data to log file
        with open(os.path.join(os.path.abspath(args.checkpoint_dir), 'logs.csv'), 'a') as f:
            f.write('{}\n'.format(','.join([str(a) for a in values])))

        # update learning rate
        lr_scheduler.step(test_results['loss'])
