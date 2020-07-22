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

parser = argparse.ArgumentParser()
parser.add_argument('input_image', default='/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg', nargs='?', action="store", type = str, help = 'Path of the image')
parser.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint_densenet121.pth', nargs='?', action="store", type = str, help='Path of the saved model to be used for prediction')
parser.add_argument('--top_k', dest="top_k", default=1, type=int, help='Number of predictions desired')
parser.add_argument('--category_names', dest="category_names", default='cat_to_name.json', help='category and class mapping')
parser.add_argument('--gpu', action='store_true', help='For using GPU for prediction')

in_arg = parser.parse_args()
input_image = ''.join(in_arg.input_image)
k = in_arg.top_k
mapping = ''.join(in_arg.category_names)
path = ''.join(in_arg.checkpoint)

if in_arg.gpu and torch.cuda.is_available():
    device = 'cuda'
elif in_arg.gpu and ~torch.cuda.is_available():
    print("GPU is not available, so we will do prediction on CPU.")
    device = 'cpu'
else:
    device = 'cpu'


model, criterion, optimizer = helper_functions.load_model(path)
# print("Model is loaded successfully")
with open(mapping, 'r') as json_file:
    cat_to_name = json.load(json_file)

prob, classes = helper_functions.predict(input_image, model, k, device)
# print("Prediction is done successfully")
category = [cat_to_name[str(cls)] for cls in classes]

for i in range(k):
    print("Rank {}: Predicted flower category {} with a probability of {}.".format(i+1, category[i], prob[i]))
    
# print("Prediction Completed")