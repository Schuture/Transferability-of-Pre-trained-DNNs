import time
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torch.optim as optim

from datasets import ImageDataset  # For image datasets other than CIFAR


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # ============================ step 1/5 construct data loader =====================
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(size=img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.225,0.225,0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.225,0.225,0.225])
    ])

    train_data = datasets.CIFAR10(root=args.data_dir, train=True, transform=train_transform)
    valid_data = datasets.CIFAR10(root=args.data_dir, train=False, transform=valid_transform)
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size)

    # ============================ step 2/5 define the model ============================
    net = models.resnet50(pretrained=False)
    net.load_state_dict(torch.load(args.a))
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes[args.dataset])
    net = net.to(device)

    # ============================ step 3/5 define the loss function ====================
    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 define the optimizer ========================
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=args.max_epoch,
                                                     eta_min=0,
                                                     last_epoch=-1)

    # ============================ step 5/5 train the model =============================
    print('\nTraining start!\n')
    start = time.time()
    max_acc = 0.
    reached = 0  # which epoch reached the max accuracy

    for epoch in range(1, args.max_epoch + 1):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()

            # print log
            loss_mean += loss.item()
            if (i+1) % args.log_interval == 0:
                loss_mean = loss_mean / args.log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, args.max_epoch, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.
                
        scheduler.step()
        
        # validate the model
        if epoch % args.val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    # forward
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    
                    # calculate the results
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).cpu().squeeze().sum().numpy()
                    loss_val += loss.item()
                    
                # record the best result
                acc = correct_val / total_val
                if acc > max_acc:
                    max_acc = acc
                    reached = epoch
                    best_model = copy.deepcopy(net)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}\n".format(
                    epoch, args.max_epoch, j+1, len(valid_loader), loss_val, acc))

    torch.save(best_model.state_dict(), args.b)
    print('\nThe model has been saved as {}'.format(args.b))
    print('\nTraining finish, the time consumption of {} epochs is {}s\n'.format(args.max_epoch, round(time.time() - start)))
    print('The max validation accuracy is: {:.2%}, reached at epoch {}.\n'.format(max_acc, reached))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tuning')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('-a', type=str, default='', help='Model pretrained on task A')
    parser.add_argument('-b', type=str, default='', help='Model finetuned on task B')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\nRunning on:", device)

    if device == 'cuda':
        device_name = torch.cuda.get_device_name()
        print("The device name is:", device_name)
        cap = torch.cuda.get_device_capability(device=None)
        print("The capability of this device is:", cap, '\n')
    
    set_seed(args.seed)
    print('random seed:', args.seed)
    
    num_classes = {'CIFAR-10': 10, 'CIFAR-100': 100, 'Caltech-101': 102,
                   'CUB-200': 200, 'Aircraft': 100, 'Flowers': 17,
                   'Land-Use': 17, 'POCUS': 3, 'DTD': 47, 'DomainNet': 345}
    if args.dataset not in num_classes.keys():
        raise ValueError('--dataset must be among {}'.format(num_classes.keys()))
    
    img_size = 32 if args.dataset.startswith('CIFAR') else 224
    
    main()









