from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark=True
plt.ion()   # interactive mode

#####################
# Training Flags #
#####################
batch_sz = 32
num_epoch = 15
init_learning_rate = 0.0001
learning_rate_decay_factor = 0.5
num_epochs_decay = 2

#####################
# data path #
#####################
data_dir = 'Data/sub' #original ISIC traning and testing data
# data_dir = 'Data/cycleGAN_data/MoleMapFalseB2000_20prc'
# data_dir = 'Data/cycleGAN_data/MoleMapFalseB2000_50prc'
# data_dir = 'Data/cycleGAN_data/MoleMapFalseB4000_20prc'
# data_dir = 'Data/cycleGAN_data/MoleMapFalseB4000_50prc'
# data_dir = 'Data/cycleGAN_data/MoleMap_20prc'
# data_dir = 'Data/cycleGAN_data/MoleMap_50prc'

######################################################################
# Load Data
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sz,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
class_num = len(class_names)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


######################################################################
# Training the model
# ------------------

def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_each_cls = [0.0] * class_num

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            class_correct = list(0 for i in range(class_num))
            class_total = list(0 for i in range(class_num))
            tmp =0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (preds[i].item() == label)
                    class_total[label] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            val_acc_each_class = [class_correct[i] / class_total[i] for i in range(class_num)]
            for i in val_acc_each_class:
                print('{:.4f}'.format(i), end=' ')

            print()
            if phase == 'val':
                # deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_acc_each_cls = val_acc_each_class
                    best_model_wts = copy.deepcopy(model.state_dict())

    # print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print(*best_acc_each_cls)

    # load best model weights
    model.load_state_dict(best_model_wts)
    # return model,best_acc, best_acc_each_cls
    return model


model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,class_num)
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
pm = model_ft.parameters()

optimizer_ft = torch.optim.Adam(model_ft.parameters(),lr = init_learning_rate)
# Decay LR by a factor of learning_rate_decay_factor every num_epochs_decay epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs_decay, gamma=learning_rate_decay_factor)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
model_ft= train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=num_epoch)
torch.save(model_ft.state_dict(), 'Molemap2ISIC.ckpt')

