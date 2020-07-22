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

def load_data(image_folder_path  = "./flowers" ):
    data_dir = image_folder_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(p=0.2),
                                           transforms.RandomVerticalFlip(p=0.2),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir ,transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir ,transform = test_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    return train_data, valid_data, test_data

def prepare_dataloaders(train_data, valid_data, test_data):
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)   
    return trainloader , validloader, testloader

    
def model_arch(cnn_arch = 'vgg16', hidden_units = 256, dropout = 0.4, lr = 0.001, device = 'cuda'):
    if cnn_arch == 'vgg16':
        cnn_model = models.vgg16(pretrained=True)
    elif cnn_arch == 'densenet121':
        cnn_model = models.densenet121(pretrained=True)
    else:
        print("Please select either vgg16 or densenet121 as CNN architecture".format(cnn_arch))
        
    for param in cnn_model.parameters():
        param.requires_grad = False
    
    input_layer = {'vgg16':25088, 'densenet121': 1024}
    classifier = nn.Sequential(OrderedDict([
        ('dropout1',nn.Dropout(dropout)),
        ('input', nn.Linear(input_layer[cnn_arch], 1024)),
        ('relu1', nn.ReLU()),
        ('dropout2',nn.Dropout(dropout)),
        ('hidden_layer1', nn.Linear(1024, hidden_units)),
        ('relu2',nn.ReLU()),
        ('hidden_layer2',nn.Linear(hidden_units,102)),
        ('output', nn.LogSoftmax(dim=1))]))

    cnn_model.classifier = classifier    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(cnn_model.classifier.parameters(), lr)    
    
    if torch.cuda.is_available() and device == 'cuda':
        cnn_model.cuda()
    return cnn_model, criterion, optimizer


def training(model, criterion, optimizer, epochs, trainloader, validloader, device):
    steps = 0
    running_loss = 0
    print_every = 40
    if torch.cuda.is_available() and device == 'cuda':
        model.cuda()
    
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            start_time = time.time()
            steps += 1
            if torch.cuda.is_available() and device=='cuda':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy=0

                for ii, (images, labels) in enumerate(validloader):
                    optimizer.zero_grad()
                    
                    if torch.cuda.is_available() and device=='cuda':
                        images, labels = images.to('cuda') , labels.to('cuda')
                    model.to(device)
                    with torch.no_grad():    
                        outputs = model.forward(images)
                        test_loss = criterion(outputs,labels)
                        prob = torch.exp(outputs).data
                        equality = (labels.data == prob.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                test_loss = test_loss / len(validloader)
                accuracy = accuracy /len(validloader)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss {:.4f}".format(test_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy),
                      "Time spent: {:.3f} seconds".format(time.time() - start_time))

                running_loss = 0
                model.train()


def save_checkpoint(path, train_data, cnn_arch, model, optimizer, hidden_units, dropout, lr, epochs):
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'cnn_arch' : cnn_arch,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                'epochs':epochs,
                'features': model.features,
                'classifier': model.classifier,
                'optim': optimizer,
                'optimizer': optimizer.state_dict(),
                'dropout': dropout,
                'lr': lr,
                'hidden_units': hidden_units
               },
               path)


def load_model(path):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(path, map_location=map_location)
    model, criterion, optimizer = model_arch(cnn_arch = checkpoint['cnn_arch'], \
                                             hidden_units = checkpoint['hidden_units'], \
                                             dropout = checkpoint['dropout'], \
                                             lr = checkpoint['lr'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])    
    return model, criterion, optimizer


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''   
    
    original_image = Image.open(str(image_path))
   
    test_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transformed_image = test_transformations(original_image)
    
    return transformed_image

def predict(image_path, model, k, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image.unsqueeze_(0)
    image = image.float()    
    if torch.cuda.is_available() and device=='gpu':
        model.to('cuda:0')
        image.cuda()
    else:
        model.to('cpu')
    
    with torch.no_grad():
        outputs = model.forward(image)
        probs, labels = torch.exp(outputs).topk(k)
        mapping = {v: k for k, v in model.class_to_idx.items()}
        keys = labels.tolist()[0]
        return probs.tolist()[0], [mapping[labels] for labels in keys]