#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:40:09 2025

@author: betsa
"""
# basics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms

batch_size = 50

# training data
#CIFAR-10 images are 32×32, 
#ImageNet network expects 224×224 input
# transform the CIFAR-10 images into 224*224

# Training data transformation  
train_data_transform = transforms.Compose([  
    transforms.Resize((224, 224)),  # Resize should take a tuple (height, width)  
    transforms.ToTensor(),  # Convert PIL Image to Tensor  
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435, 0.2616))  # Normalize after converting to Tensor  
])  

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  
                                         download=True, transform=train_data_transform)    
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)                                         
                                           
# Validation data
val_data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4821, 0.4465), (0.2470, 0.2435,0.2616))
])
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True,transform=val_data_transform)

val_order = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

# Choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Training of the model
def train_model(model, loss_function, optimizer, data_loader):
  # set model to training mode
  model.train()
  current_loss = 0.0
  current_acc = 0
  # iterate over the training data
  for i, (inputs, labels) in enumerate(data_loader):
      # send the input/labels to the GPU
      inputs = inputs.to(device)
      labels = labels.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()
      with torch.set_grad_enabled(True):
      # forward
          outputs = model(inputs)
          _, predictions = torch.max(outputs, 1)
          loss = loss_function(outputs, labels)
      # backward
          loss.backward()
          optimizer.step()
  # statistics
      current_loss += loss.item() * inputs.size(0)
      current_acc += torch.sum(predictions == labels.data)
  total_loss = current_loss / len(data_loader.dataset)
  total_acc = current_acc.double() / len(data_loader.dataset)
  print(total_acc)
  print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss,total_acc))
  
   
          
# testing/validation     
def test_model(model, loss_function, data_loader):
  # set model in evaluation mode
  model.eval()
  current_loss = 0.0
  current_acc = 0  
  # iterate over the validation data
  for i, (inputs, labels) in enumerate(data_loader):
      # send the input/labels to the GPU
      inputs = inputs.to(device)
      labels = labels.to(device)
      # forward
      with torch.set_grad_enabled(False):
          outputs = model(inputs)
          _, predictions = torch.max(outputs, 1)
          loss = loss_function(outputs, labels)
      # statistics
      current_loss += loss.item() * inputs.size(0)
      current_acc += torch.sum(predictions == labels.data)
  total_loss = current_loss / len(data_loader.dataset)
  total_acc = current_acc.double() / len(data_loader.dataset)
  print('Test Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss,total_acc))
  return total_loss, total_acc
          
## ____________________________________________________________________________
## Transfer learning in 4 steps:
    #1- use ResNet-18 and automatically download the pretrained weights.
    #2- Replace the last network layer with a new layer with 10 outputs 
    #3- Exclude the existing network layers from the backward pass and only
        #pass the newly added fully-connected layer to the Adam optimizer.
    #4- Run the training for epochs and evaluate the network accuracy a

#  training new added fully conected layer
def tl_feature_extractor(epochs=5):
   # load the pretrained model
   model = torchvision.models.resnet18(pretrained=True)
   
   # exclude existing parameters from backward pass
   # for performance
   for param in model.parameters():
     param.requires_grad = False
   
   # newly constructed layers have requires_grad=True by default
   num_features = model.fc.in_features
   model.fc = nn.Linear(num_features, 10)     
   
   # transfer to GPU (if available)
   model = model.to(device)
   loss_function = nn.CrossEntropyLoss()
   # only parameters of the final layer are being optimized
   optimizer = optim.Adam(model.fc.parameters())     
   
   # train
   test_acc = list() # collect accuracy for plotting
   for epoch in range(epochs):
     print('Epoch {}/{}'.format(epoch + 1, epochs))
     train_model(model, loss_function, optimizer, train_loader)
     _, acc = test_model(model, loss_function, val_order)
     test_acc.append(acc)
  
   plt.plot(test_acc)
   plt.show()
   
   
#training the whole network
def tl_fine_tuning(epochs=5):
   # load the pretrained model
   model = models.resnet18(pretrained=True)
   # replace the last layer
   num_features = model.fc.in_features
   model.fc = nn.Linear(num_features, 10)
   # transfer the model to the GPU
   model = model.to(device)
   # loss function
   loss_function = nn.CrossEntropyLoss()
   # We'll optimize all parameters
   optimizer = optim.Adam(model.parameters())   
   # train
   test_acc = list() # collect accuracy for plotting
   for epoch in range(epochs):
     print('Epoch {}/{}'.format(epoch + 1, epochs))
     train_model(model, loss_function, optimizer, train_loader)
     _, acc = test_model(model, loss_function, val_order)
     test_acc.append(acc)

   plt.plot(test_acc)
   plt.show()
   

tl_feature_extractor(epochs = 5)  
tl_fine_tuning(epochs=5) 
