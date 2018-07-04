# PROGRAMMER: Volkan KUDAY
# DATE CREATED: 06/18/2018
# REVISED DATE:               <=(Date Revised - if any)
# PURPOSE: Trains a new network on a dataset of images
#
# Use argparse Expected Call with <> indicating expected user input:
#     Basic Usage: python train.py --dir <data_dir> -->Prints out: training loss, validation loss, and validation accuracy using model VGG16
#       Options:        
#                - Set directory to save checkpoints:  python train.py --dir <data_dir> --checkpoint <save_checkpoint>
#                           Example Call: python train.py --dir "flowers/" --checkpoint "checkpoint.pth"
#
#                - Choose architecture:  python train.py --dir <data_dir> --arch <model>
#                           Example Call: python train.py --dir "flowers/" --arch "vgg16"
#
#                - Set hyperparameters:  python train.py --dir <data_dir> --learning_rate <lr> --hidden_units <number_of_hidden_units> --
#                                        epochs <number_of_epochs>
#                           Example Call: python train.py --dir "flowers/" --learning_rate 0.001 --hidden_units 1024 --epochs 20
#
#                - Use GPU for training: python train.py --dir <data_dir> --gpu

# Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from time import time

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def get_input_args():
    '''
    Define command line arguments
    '''
    # Create Argument Parser object named parser
    parser = argparse.ArgumentParser(description = 'Create command line arguments for training phase')
    
    # Argument 1: That's a folder that includes flower image files (default - 'flowers/')
    parser.add_argument('--dir', type = str, default = 'flowers/',
                        help = 'path to the folder that includes flower files --> --dir "flowers/" ')
    
    # Argument 2: Save trained model using for prediction in the future
    parser.add_argument('--checkpoint', type = str,
                        help = 'save trained model checkpoint to folder --> --checkpoint "checkpoint_vgg16.pth" ')
    
    # Argument 3: That's CNN model architecture (default - 'vgg16')
    parser.add_argument('--arch', type = str, default = 'vgg16',
                        help = 'chosen CNN model architecture: vgg16, densenet121, alexnet --> --arch "vgg16" ')
    
    # Argument 4: That's learning rate for optimization during backpropagation
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                        help = 'chosen learning rate --> --learning_rate 0.001 ')
    
    # Argument 5: That's number of hidden units for hidden layer
    parser.add_argument('--hidden_units', type = str, default = '4096, 1000',
                        help = 'chosen number of hidden units --> --hidden_units "4096, 1000" ')
    
    # Argument 6: That's number of epochs
    parser.add_argument('--epochs', type = int, default = 20,
                        help = 'chosen number of epochs --> --epochs 20 ')
    
    # Argument 7: Use GPU for training
    parser.add_argument('--gpu',  action = 'store_true',
                        help = 'Use GPU for training --> --gpu ')
    
    return parser.parse_args()

def define_data(in_args):
    
    if in_args.dir:
        # Define path for train, test, and validation image files
        data_dir = in_args.dir
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # Transforms for the training, validation, and testing sets
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        data_transforms = {
            'train_transforms': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'valid_transforms': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize 
            ]),
             'test_transforms': transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 normalize
             ])
        }

        # Load the datasets with ImageFolder
        image_datasets = {
            'train_dataset': datasets.ImageFolder(train_dir, transform = data_transforms['train_transforms']),
            'valid_dataset': datasets.ImageFolder(valid_dir, transform = data_transforms['valid_transforms']),
            'test_dataset' : datasets.ImageFolder(test_dir, transform = data_transforms['test_transforms'])
        }

        # Define the dataloaders
        dataloaders = {
            'train_loader': torch.utils.data.DataLoader(image_datasets['train_dataset'], batch_size = 64, shuffle = True),
            'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_dataset'], batch_size = 32, shuffle = False),
            'test_loader' : torch.utils.data.DataLoader(image_datasets['test_dataset'], batch_size = 32, shuffle = True)
        }
        
        return image_datasets, dataloaders
    
def load_model(arch = 'vgg16', output_size = 102, hidden_layers = [4096, 1000]):
    # Load a pre-trained model
    if arch == 'vgg16':
        # Load vgg16 pre-trained model
        model = models.vgg16(pretrained = True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        # Load resnet18 pre-trained model
        model = models.densenet121(pretrained = True)
        input_size = model.classifier.in_features
    elif arch == 'alexnet':
        # Load alexnet pre-trained model
        model = models.alexnet(pretrained = True)
        input_size = model.classifier[1].in_features
    else:
        raise ValueError('"{}" network architecture is not defined. Please select "vgg16", "densenet121" or "alexnet" as a pre-trained '                                 'model'.format(arch))
        
    # Freeze parameters and define a new feedforward network for use as a classifier using the features as input
    for param in model.parameters():
        param.requires_grad = False
    
    # Create ModuleList and add input layer
    hidden_layer = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
    
    # Add a variable number of more hidden layers
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    
    # Add hidden layer to the ModuleList
    hidden_layer.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
    
    # Create fully-connected network
    fc = [hidden_layer[0], nn.ReLU(), nn.Dropout(p=0.5)]
    for i in range(len(hidden_layer)-1):
        fc.append(hidden_layer[i+1])
        fc.append(nn.ReLU())
        fc.append(nn.Dropout(p=0.5))
    fc.append(nn.Linear(hidden_layers[-1], output_size))
    fc.append(nn.LogSoftmax(dim=1))
    
    model.classifier = nn.Sequential(*fc)
    
    return model

# Avoid overfitting when training the network with using validation method
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.cuda(), labels.cuda()
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1]) 
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy

# Train model
def train_model(arch = 'vgg16', epochs = 20, learning_rage = 0.001, gpu = False, checkpoint = '', hidden_layers = [4096, 1000]):
    
    in_args = get_input_args()
    image_datasets, dataloaders = define_data(in_args)
    
    if in_args.arch:
        arch = in_args.arch
     
    if in_args.hidden_units:
        # Get hidden layers from hidden_units
        hidden_layers = [int(i) for i in in_args.hidden_units.split(',')]
        hidden_layers = hidden_layers
        
    if in_args.epochs:
        epochs = in_args.epochs
        
    if in_args.learning_rate:
        learning_rate = in_args.learning_rate
        
    if in_args.gpu:
        gpu = in_args.gpu
        
    if in_args.checkpoint:
        checkpoint = in_args.checkpoint
        
    print ('\n CNN model architecture: {}\n'.format(arch),
           'Hidden layer sizes: {}\n'.format(hidden_layers),
           'Number of epochs: {}\n'.format(epochs),
           'Learning rate: {}\n'.format(learning_rate)
          )
    
    # Load the model
    output_size = len(image_datasets['train_dataset'].classes)
    model = load_model(arch, output_size=output_size, hidden_layers=hidden_layers)
    
    # Check GPU is active
    if gpu and torch.cuda.is_available():
        print('GPU will be used for training.\n')
        device = torch.device('cuda:0')
        model.cuda()
    else:
        print('CPU will be used for training.\n')
        device = torch.device('cpu')
    
    # Defining criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
              
    start = time()
    print('Training started. Please wait until the training is finished.\n')         
    best_accuracy = 0.0
    print_every = 40
    steps = 0
    running_loss = 0
    
    for e in range(epochs):
        model.train()
        for images, labels in dataloaders['train_loader']:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
            
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, dataloaders['valid_loader'], criterion)
                
                best_accuracy =  accuracy if accuracy > best_accuracy else best_accuracy
                
                print(' Epoch: {}/{}: \n'.format(e+1, epochs),
                      'Training Loss: {:.3f}\n'.format(running_loss / print_every),
                      'Validation Loss: {:.3f}\n'.format(valid_loss / len(dataloaders['valid_loader'])),
                      'Accuracy: {:.2f}%\n'.format(100 * accuracy / len(dataloaders['valid_loader'])),
                      'Best Accuracy: {:.2f}%\n'.format(100 * best_accuracy / len(dataloaders['valid_loader'])),
                      '-' * 20)
            
                running_loss = 0
                model.train()
            
    total_time = time() - start
    
    # Prints overall runtime in format hh:mm:ss
    print('Total Elapsed Runtime:', str( int( (total_time / 3600) ) ) + ':' +
          str( int( ( (total_time % 3600) / 60 ) ) ) + ':' +
          str( int( ( (total_time % 3600) % 60 ) ) ) )
    
    model.class_to_idx = image_datasets['train_dataset'].class_to_idx
    
    # Save checkpoint
    if checkpoint:
        print('Saving checkpoint to {}'.format(checkpoint))
        checkpoint_dict = {
            'arch': arch,
            'hidden_layers': hidden_layers,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx
        }
        torch.save(checkpoint_dict, checkpoint)
        
    return model

if __name__ == '__main__':
    train_model()


           
