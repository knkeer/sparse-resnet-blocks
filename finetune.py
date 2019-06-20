import argparse
import os
import time

import tabulate
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.glt_models import LinearClassifier
from sparse_finetune.sparse_finetune import SequentialSparsifier
from training import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential sparsification of pretrained residual networks.')
    parser.add_argument('--model-path', type=str, default='./model.pth', required=False,
                        help='path to checkpoint with pretrained model (default: ./model)')
    parser.add_argument('--data-dir', type=str, default='./cifar10', required=False,
                        help='path to dataset fdirectory (default: ./cifar10`0')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints-ssf', required=False,
                        help='path to checkpoint directory (default: ./checkpoints-ssf)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, required=False,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--adam', action='store_true', required=False,
                        help='use ADAM optimizer instead of SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-4, required=False, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--num-epochs', type=int, default=200, required=False,
                        help='number of finetuning epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128, required=False,
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, required=False,
                        help='number of workers (default: 4)')
    parser.add_argument('--seed', type=int, default=12345, required=False,
                        help='random seed (default: 12345)')
    parser.add_argument('--save-freq', type=int, default=25, required=False, metavar='N',
                        help='number of epochs between checkpoints (default: 25)')
    parser.add_argument('--device', type=str, default=None, required=False,
                        help="device to train on (default: 'cuda:0' if torch.cuda.is_available() else 'cpu')")
    parser.add_argument('--depth', type=int, default=110, required=False, metavar='N',
                        help='depth of residual network (default: 110)')
    parser.add_argument('--reverse', action='store_true', required=False,
                        help='reverse layer order of residual network (default: off)')
    parser.add_argument('--width', type=int, default=1, required=False,
                        help='width of residual network (default: 1)')
    parser.add_argument('--num-blocks', type=int, default=6, required=False,
                        help='number of blocks that are sparsified at a time (default: 6)')
    args = parser.parse_args()
    args.sparse = False

    utils.set_random_seed(args.seed)

    print('Preparing checkpoint folder: {}'.format(os.path.abspath(args.checkpoint_dir)))
    utils.prepare_checkpoint_folder(args)

    print('Loading CIFAR-10 from {}'.format(os.path.abspath(args.data_dir)))
    dataloader = utils.get_dataloader(args)

    layers = utils.get_resnet_layers(args)
    linear_classifier = LinearClassifier(layers[-1].out_channels, sparse=args.sparse)
    model = nn.Sequential(*layers, linear_classifier)

    if not os.path.isfile(args.model_path):
        raise ValueError('File {} does not exist'.format(os.path.abspath(args.model_path)))
    else:
        print('Loading model from {}'.format(os.path.abspath(args.model_path)))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
    train_mask = [False for _ in range(len(model))]
    model = SequentialSparsifier(model)

    print('Using {} optimizer with initial learning rate {} and weight decay {}'.format(
        'ADAM' if args.adam else 'SGD', args.learning_rate, args.weight_decay
    ))

    # set loss function
    criterion = nn.NLLLoss()

    # prepare logs
    columns = ['block', 'epoch', 'lr', 'tr_ce', 'tr_acc', 'te_ce', 'te_acc', 'kl', 'sp', 'beta', 'time']
    fmt = ['%s', '%s', '3.1e', '8.6f', '5.2f', '8.6f', '5.2f', '8.1f', '6.2f', '4.2f', '5.1f']
    with open(os.path.join(args.checkpoint_dir, 'logs.csv'), 'w') as f:
        f.write('{}\n'.format(','.join(columns)))

    # get device to train on
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for start_idx in range(0, len(train_mask), args.num_blocks):
        # select layers for training
        train_mask = [False for _ in range(len(train_mask))]
        for i in range(start_idx, min(len(train_mask), start_idx + args.num_blocks)):
            train_mask[i] = True

        # update train mask for model and get optimizer
        model.update_mask(train_mask)
        optimizer = utils.get_optimizer(model, args)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5)

        # create block checkpoint directory
        block_checkpoint_dir = os.path.join(args.checkpoint_dir, 'block-{}'.format(start_idx // args.num_blocks + 1))
        os.makedirs(block_checkpoint_dir, exist_ok=True)

        for epoch in range(args.num_epochs):
            beta = min(0.02 * epoch, 1.0)
            epoch_time = time.time()
            train_results = utils.train_model(model, optimizer, criterion, dataloader['train'], device=device,
                                              beta=beta, finetune=True)
            test_results = utils.evaluate_model(model, criterion, dataloader['test'], device=device, beta=beta)
            epoch_time = time.time() - epoch_time

            # save model if necessary
            if (epoch + 1) % args.save_freq == 0 or epoch == args.num_epochs - 1:
                utils.make_checkpoint(
                    block_checkpoint_dir,
                    epoch + 1,
                    block=start_idx // args.num_blocks + 1,
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    lr_scheduler=lr_scheduler.state_dict()
                )

            # get current learning rate
            current_lr = args.learning_rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']

            # get data about this training epoch
            values = [start_idx // args.num_blocks + 1, epoch + 1, current_lr, train_results['loss'],
                      train_results['accuracy'], test_results['loss'], test_results['accuracy'],
                      test_results['kl'], beta, epoch_time]
            total, non_zero = 0, 0
            for module, train_flag in zip(model.model, train_mask):
                if train_flag:
                    module_total, module_non_zero = utils.parameter_statistics(module)
                    total += module_total
                    non_zero += module_non_zero
            compression = float('inf') if non_zero == 0 else total / non_zero
            values = values[:-2] + [compression] + values[-2:]

            # print data to standard output
            table = tabulate.tabulate([values], columns, floatfmt=fmt)
            if epoch % 40 == 0:
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

        model.finalize_blocks(train_mask)

    # show final compression of final model and save it
    final_model = nn.Sequential(*model.model)
    test_results = utils.evaluate_model(final_model, criterion, dataloader['test'], device=device)
    print('Final compression ratio: {}'.format(utils.compression_ratio(final_model)))
    print('Test accuracy: {}'.format(test_results['accuracy']))
    torch.save({'model': final_model.state_dict(),
                'compression': utils.compression_ratio(final_model),
                'accuracy': test_results['accuracy']},
               os.path.join(args.checkpoint_dir, 'result.pth'))
