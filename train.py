import numpy as np
import json
import time
import argparse
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import helper_functions

parser = argparse.ArgumentParser(description='Training CNN Model')

# Command Line arguments

parser.add_argument('data_dir', nargs=1, default="/home/workspace/ImageClassifier/flowers/", help='Images Folder')
parser.add_argument('--save_dir', dest="save_dir", default="/home/workspace/ImageClassifier/checkpoint.pth", help='path where the model is to be saved post training')
parser.add_argument('--learning_rate', dest="learning_rate", default=0.001, help='Learning rate for Adam optimizer.')
parser.add_argument('--dropout', dest = "dropout", default = 0.4, help='Dropout to control regularization of the model')
parser.add_argument('--epochs', dest="epochs", default=1, type=int, help='Number of epochs for which the model is to be trained')
parser.add_argument('--arch', dest="arch", default="densenet121", type = str, help='which pretrained model is to be used for training. Please choose between densenet121 and vgg16.')
parser.add_argument('--hidden_units', dest="hidden_units", default=256, type=int, help='Number of hidden units in the last hidden layer')
parser.add_argument('--gpu', action='store_true', help='To use GPU for training')

in_arg = parser.parse_args()

image_folder_path = ''.join(in_arg.data_dir)
path = ''.join(in_arg.save_dir)
lr = in_arg.learning_rate
cnn_arch = in_arg.arch
dropout = in_arg.dropout
hidden_units = in_arg.hidden_units
epochs = in_arg.epochs

if in_arg.gpu and torch.cuda.is_available():
    device = 'cuda'
elif in_arg.gpu and ~torch.cuda.is_available():
    print("GPU is not available, so prediction will be done on CPU.")
    device = 'cpu'
else:
    device = 'cpu'

#Loading datasets for model training
train_data, valid_data, test_data = helper_functions.load_data(image_folder_path)
trainloader , validloader, testloader = helper_functions.prepare_dataloaders(train_data, valid_data, test_data)

model, criterion, optimizer = helper_functions.model_arch(cnn_arch, hidden_units, dropout, lr, device)
print("Model training has started")
helper_functions.training(model, criterion, optimizer, epochs, trainloader,validloader,device)
print("Model has been trained successfully")
helper_functions.save_checkpoint(path, train_data, cnn_arch, model, optimizer, hidden_units, dropout, lr, epochs)
print("Model Training is completed and Final model is saved for future use. Thanks Udacity!")