# PROGRAMMER: Volkan KUDAY
# DATE CREATED: 06/19/2018
# REVISED DATE:               <=(Date Revised - if any)
# PURPOSE: Predict flower name from an image
#
# Use argparse Expected Call with <> indicating expected user input:
#     Basic Usage: python predict.py --path <image_path> --checkpoint <checkpoint> -->Prints out: flower name and class probability
#       Options:        
#                - Return Top K most likely classes: python predict.py --path <image path> --checkpoint <checkpoint> --top_k <top k>
#                           Example Call: python predict.py --path "data/img_1622.jpg" --checkpoint "checkpoint_densenet121.pth" --top_k 5
#
#                - A mapping of categories to real names: python predict.py --path <image path> --checkpoint <checkpoint> --cat_to_name   
#                                                         <category to real name>
#                           Example Call: python predict.py --path "data/img_1622.jpg" --checkpoint "checkpoint_densenet121.pth" 
#                                         --cat_to_name "/cat_to_name.json"
#
#                - Use GPU for inference: python predict.py --path <image path> --checkpoint <checkpoint> --gpu
#                           Example Call: python predict.py --path "data/img_1622.jpg" --checkpoint "checkpoint_densenet121.pth" --gpu

# Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

import torch

from train import load_model
from PIL import Image

def get_input_args():
    '''
    Define command line arguments
    '''
    # Create Argument Parser object named parser
    parser = argparse.ArgumentParser(description = 'Create command line arguments for inference phase')
    
    # Argument 1: That's a single image path (default - 'flowers/test/34/image_06941.jpg')
    parser.add_argument('--image_path', type = str, default = 'flowers/test/34/image_06941.jpg',
                        help = 'path to the image that will be predicted --> --path "flowers/test/34/image_06941.jpg" ')
    
    # Argument 2: Load trained model using for inference
    parser.add_argument('--checkpoint', type = str,
                        help = 'load trained model using for inference --> --checkpoint "checkpoint.pth" ')
    
    # Argument 3: Return Top K most likely classes (default - 5)
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'return top K most likely classes --> --top_k 5 ')
    
    # Argument 4: Mapping of categories to real names
    parser.add_argument('--cat_to_name', type = str,
                        help = 'mapping of categories to real names (json file) --> --cat_to_name "cat_to_name.json" ')
    
    # Argument 5: Use GPU for inference
    parser.add_argument('--gpu', action = 'store_true',
                        help = 'Use GPU for inference --> --gpu ')
    
    return parser.parse_args()

# Label mapping
def label_mapping(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
        
# Load the checkpoint and rebuild the model
def load_checkpoint(checkpoint_path):
    checkpoint_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    arch = checkpoint_dict['arch']
    hidden_layers = checkpoint_dict['hidden_layers']
    output_size = len(checkpoint_dict['class_to_idx'])
    
    model = load_model(arch = arch, output_size = output_size, hidden_layers = hidden_layers)
    
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
 
    return model
    
# Image preprocessing for input to model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model using PIL library
    '''  
    image_prep = Image.open(image)
    
    # Keeping the aspect ratio
    width_size, height_size = image_prep.size
    ratio = height_size / width_size
    grab_size = height_size if height_size < width_size else width_size
    new_width_size, new_height_size = 0, 0
    if grab_size == height_size:
    	new_height_size = 256
    	new_width_size = int(new_height_size / ratio)
    else:
    	new_width_size = 256
    	new_height_size = int(ratio * new_width_size)
	    
    # Resize the image where the shortest side is 256 pixels
    resized_image = image_prep.resize((new_width_size, new_height_size), Image.ANTIALIAS)
    
    # Crop the image to 224x224
    new_width, new_height = 224, 224
    width, height = resized_image.size
    dimension = {
        'left' : (width - new_width) / 2,
        'right' : (width + new_width) / 2,
        'top' : (height - new_height) / 2,
        'bottom' : (height + new_height) / 2
    }
    cropped_image = resized_image.crop(( dimension['left'], dimension['top'], dimension['right'], dimension['bottom'] ))
    
    # Model expected floats 0-1. Convert color channel values of images to floats 0-1
    converted_image = np.array(cropped_image) / 255
    
    # Normalized the image properly for the model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (converted_image - mean) / std
    
    # Reorder process: The color channel is the third dimension in the PIL image and Numpy array (224, 224, 3), 
    # Pytorch expects the color channel to be the first dimension (3, 224, 224)
    image = normalized_image.transpose((2,0,1))
    
    # Convert numpy array to pytorch tensor
    return torch.from_numpy(image)

# Preparing the image for display
def imshow(image, ax = None, title = None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
        
    # Pytorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image, (1, 2, 0)).data.numpy()
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = image * std + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    original_image = np.clip(original_image, 0, 1)
    
    ax.imshow(original_image)
    return ax

# Predict the class of the flower in the image
def predict(model, in_args, image = 'flowers/test/34/image_06941.jpg', top_k = 5, cat_to_name = '',  gpu = False):
    
    if in_args.image_path:
        image = in_args.image_path
       
    if in_args.top_k:
        top_k = in_args.top_k
    
    if in_args.cat_to_name:
        cat_to_name = label_mapping(in_args.cat_to_name)     
    
    if in_args.gpu:
        gpu = in_args.gpu
        
    # Check GPU is active
    if gpu and torch.cuda.is_available():
        print('GPU will be used for prediction.\n')
        device = torch.device('cuda:0')
        model.cuda()
    else:
        print('CPU will be used for prediction.\n')
        device = torch.device('cpu')
        
    # Evaluate mode --> dropout is turned off
    model.eval()
    
    # Image preprocessing for input to model
    input_image = process_image(image)
    input_image = input_image.unsqueeze(0).float()
    
    input_image = input_image.to(device)
    
    with torch.no_grad():
        output = model.forward(input_image)
    
    ps = torch.exp(output)
    
    probs, classes = ps.topk(top_k)
    probs = [prob for prob in probs.cpu().data.numpy().squeeze()]
    classes = [label for label in classes.cpu().data.numpy().squeeze()]
    
    idx_to_class = {model.class_to_idx[i]:i for i in model.class_to_idx.keys()}
    
    labels = []
    if cat_to_name:
        for i in range(len(classes)):
            labels.append(cat_to_name[idx_to_class[classes[i]]])
    
    print('Inference Done.\n\n'
          'It has been predicted as "{}" with {:.2f}%\n'
          .format(labels[0] if labels else classes[0], probs[0]*100))
    
    fig, (ax1, ax2) = plt.subplots(figsize = (9, 9), ncols = 2)
    imshow(process_image(image), ax1)
    
    # Image axes settings
    ax1.axis('off')
    ax1.set_title(labels[0] if labels else classes[0])
    
    # Class probability axes settings
    ax2.set_title('Class Probability')
    ax2.barh(np.arange(len(probs)), probs)
    ax2.set_aspect(0.2)
    ax2.set_yticks(np.arange(len(labels if labels else classes)))
    ax2.set_yticklabels(labels if labels else classes)
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig('prediction.png')
    plt.show()
    
def main():
    # Get arguments from user
    in_args = get_input_args()
   
    # Checkpoint file is being checked
    if in_args.checkpoint:
        checkpoint = in_args.checkpoint
        model = load_checkpoint(checkpoint)
        predict(model, in_args)
    else:
        print('Please specify path of checkpoint file that will be used for rebuilding the model')
        
if __name__ == '__main__':
    main()
