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
from PIL import Image
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

####################################################################
## data preprocessing and loading
####################################################################
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#####################
## MoleMap
#####################
# Molemap_dir ='MoleMap/classes' # data path #
Molemap_dir ='Data/cycleGAN_data/MoleMapFalseB2000_N_ISIC_Train'
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms_m = {
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

image_datasets_m = {x: datasets.ImageFolder(os.path.join(Molemap_dir, x),
                                          data_transforms_m[x])
                  for x in ['train', 'val']}
dataloaders_m = {x: torch.utils.data.DataLoader(image_datasets_m[x], batch_size=batch_sz,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes_m = {x: len(image_datasets_m[x]) for x in ['train', 'val']}
class_names_m = image_datasets_m['train'].classes


#####################
## ISIC
#####################
data_dir = 'Data/cycleGAN_data/2000Balanced'  # data path #
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Training the model
# ------------------


def train_model(model, criterion, optimizer, scheduler,dataloaders, data_sizes, num_epochs=num_epoch):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_train=[]
    acc_train=[]
    loss_val=[]
    acc_val=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

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

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'train':
                loss_train.append(epoch_loss)
                acc_train.append(epoch_acc)
            if phase == 'val':
                loss_val.append(epoch_loss)
                acc_val.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    result=[loss_train,acc_train,loss_val,acc_val]

    return model,result,time_elapsed


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Finetuning the convnet transfer from resnet to MoleMap
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
model_ft = models.resnet152(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names_m))
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
model_ft.parameters()
# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=init_learning_rate)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs_decay, gamma=learning_rate_decay_factor)

######################################################################
# Train and evaluate and save model
model_ft,result,time_elapsed = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders_m,dataset_sizes_m,
                       num_epochs=num_epoch)
torch.save(model_ft.state_dict(), 'Molemap.ckpt')
print(result)
print(time_elapsed)
######################################################################
# Finetuning the convnet transfer from MoleMap to ISIC
# ----------------------
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)
model_ft.parameters()
# Observe that all parameters are being optimized
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=init_learning_rate)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs_decay, gamma=learning_rate_decay_factor)
model_ft,result,time_elapsed = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,dataset_sizes,
                       num_epochs=num_epoch)
print(result)
print(time_elapsed)
######################################################################


