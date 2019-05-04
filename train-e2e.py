from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from sparse_resnet import models, utils

if __name__ == '__main__':
    compute_compession = False
    cifar_train = utils.get_cifar(batch_size=128, train=True)
    cifar_test = utils.get_cifar(batch_size=128, train=False)
    model = models.make_resnet(164, sparse=False)
    optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.NLLLoss()
    num_epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_name = 'resnet-164-dense-e2e'
    
    utils.create_dir(experiment_name)
    f = open(experiment_name + '/logs.csv', 'w+')
    if compute_compession:
        f.write('epoch,train_loss,train_accuracy,test_loss,test_accuracy,compression\n')
    else:
        f.write('epoch,train_loss,train_accuracy,test_loss,test_accuracy\n')

    model.to(device)
    model.train()
    for epoch in range(1, num_epochs + 1):
        # train for epoch and collect average train loss and acc
        train_loss, correct = 0.0, 0
        for samples, targets in tqdm(cifar_train):
            samples, targets = samples.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(samples)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += torch.sum(torch.argmax(logits, dim=-1) == targets).item()
        # validate and update logs
        test_loss, test_accuracy = utils.validate(model, criterion, cifar_test, device=device)
        train_loss /= len(cifar_train)
        train_accuracy = correct / len(cifar_train.dataset)
        if compute_compession:
            compression = utils.compute_compession(model)
            f.write('{},{},{},{},{},{}\n'.format(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy, compression
            ))
            print('epoch: {:d} | tr_loss: {:1.03f} | tr_acc: {:2.01f}% | te_loss: {:1.03f} | te_acc: {:2.01f}% | comp: {:.01f}'.format(
                epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100, compression
            ))         
        else:
            f.write('{},{},{},{},{}\n'.format(
                epoch, train_loss, train_accuracy, test_loss, test_accuracy
            ))
            print('epoch: {:d} | tr_loss: {:1.03f} | tr_acc: {:2.01f}% | te_loss: {:1.03f} | te_acc: {:2.01f}%'.format(
                epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100
            )) 
        utils.checkpoint(str(epoch), model, optimizer, experiment_name)
        # update learning rate if necessary
        lr_scheduler.step(test_loss)            
    