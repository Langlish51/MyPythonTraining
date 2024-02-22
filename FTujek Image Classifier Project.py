#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


#Notes
  
    #reformat the pictures & Create a loader:"Loading Image Data" in "Deep Learning with PyTorch" video
    #Train model: see "Transfer Learning II" "Deep Learning with PyTorch" video
    #Test model:check "Transfer Learning Solution" video


# In[44]:


# My imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from IPython.display import Image
from PIL import Image as PILImage
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[1]:


# Define my path (all folders and files at current directory)
data_dir = './flower_data'
train_dir = f'{data_dir}/train'
valid_dir = f'{data_dir}/valid'
test_dir = f'{data_dir}/test'


# In[3]:


# DONE: Define my transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
}

# DONE: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),}

# DONE: Using the image datasets and the transforms, define my loaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'valid': DataLoader(image_datasets['valid'], batch_size=32),
    'test': DataLoader(image_datasets['test'], batch_size=32),}


# In[5]:


# DONE: Test my dataloaders

numbers_train = len(dataloaders['train'].dataset)
print(numbers_train)

numbers_valid = len(dataloaders['valid'].dataset)
print(numbers_valid)

numbers_test = len(dataloaders['test'].dataset)
print(numbers_test)


# In[6]:


# DONE: Check the size of my dataset
num_classes = len(image_datasets['train'].class_to_idx)
print(num_classes)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[19]:


import json

# DONE: Load the label mapping from my directory
file_path = 'cat_to_name.json'

with open(file_path, 'r') as f:
    cat_to_name = json.load(f)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[8]:


# DONE: Build and train my network

# Load VGG16 model
model = models.vgg16(pretrained=True)


# In[9]:


# Check my model loaded correctly
print(model)


# In[10]:


# Freezing VGG16 original parameters
for original_param in model.parameters():
    original_param.requires_grad = False


# In[2]:


# Creating a custom classifier (initial standard classifier returned low accuracy around 6%)

class CustomClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add some more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers` with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

# Match the VGG16's classifier input size
input_size = 25088
output_size = 102  # Number of classes in your dataset
hidden_layers = [4096, 2048]  # Define the sizes of the hidden layers
drop_p = 0.5  # Dropout probability

# Create an instance of the custom classifier
model.classifier = CustomClassifier(input_size, output_size, hidden_layers, drop_p)


# In[12]:


# Check my new classifier
print(model.classifier)


# In[13]:


# Select GPU or CPU (Udacity GPU doesn't work!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Write the Loss function
criterion = nn.NLLLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Update to optimize all model parameters

# Set the number of epochs
epochs = 6
for epoch in range(epochs):
    model.train()  # Set to training mode
    running_loss = 0
    
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Perform a single optimization step
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation
    model.eval()  # Set to evaluation mode
    validation_loss = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
    
    # Print out the average loss per epoch
    print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(dataloaders['train']):.3f} - Validation loss: {validation_loss/len(dataloaders['valid']):.3f}")


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[14]:


#DONE: Test my model

# evaluation mode
model.eval()

# My variables
correct = 0
total = 0

# Go through the test dataset
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        # Move data to the appropriate device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Probabilities
        ps = torch.exp(outputs)
        
        # Top class and probability per image
        top_p, top_class = ps.topk(1, dim=1)
        
        # Check predicted class matches the true labels
        equals = top_class == labels.view(*top_class.shape)
        
        # Increment correct count 
        correct += equals.sum().item()
        
        # Increment total count
        total += labels.size(0)

# Calculate accuracy
accuracy = correct / total

# Print the accuracy of my model
print(f"Test Accuracy: {accuracy:.3f}")


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[15]:


# Create dictionary
checkpoint = {
    'model_state_dict': model.state_dict(),  # This includes the classifier state if present
    'class_to_idx': image_datasets['train'].class_to_idx,  # Save class to index mapping
    'num_classes': len(image_datasets['train'].classes),  # Add number of classes
    'epochs': epochs,  # Number of epochs trained
    'optimizer_state_dict': optimizer.state_dict()  # Save optimizer state
}

# Save checkpoint with the name 'Checkpoint3.pth'
checkpoint_path = 'Checkpoint3.pth'
torch.save(checkpoint, checkpoint_path)

print("Checkpoint saved successfully as Checkpoint3.pth.")


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[45]:


# Define my CustomClassifier
class CustomClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super(CustomClassifier, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

# Instantiate my model, replace the classifier, load the checkpoint
model = models.vgg16(pretrained=False)

# Set the custom classifier
input_size = 25088  # Match the input size of VGG
output_size = 102  # Number of flower classes
hidden_layers = [4096, 2048]  # My hidden layers setup
model.classifier = CustomClassifier(input_size, output_size, hidden_layers)

# Load the checkpoint
checkpoint_path ='Checkpoint3.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()  # Set the model to evaluation mode

print("Checkpoint loaded successfully.")


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[46]:


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.
    '''
    # Open the image using PILImage
    img = PILImage.open(image_path)
    
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    img_tensor = preprocess(img)
    
    return img_tensor  # Return the processed tensor


# In[ ]:


To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


# In[47]:


# Process the image and store the result in processed_image
processed_image = process_image('Rosepicture.jpg')

# Print processed image details
print(processed_image.shape)
print(processed_image[:1, :1, :])  # Print the first pixel values


# In[48]:


def imshow(tensor, ax=None):
    """Display a PyTorch tensor as an image."""
    if ax is None:
        fig, ax = plt.subplots()

    # Convert PyTorch tensor to numpy array
    image = tensor.cpu().numpy().squeeze()  # Remove batch dimension and transfer to CPU

    # Transpose from (C, H, W) to (H, W, C) and undo normalization
    image = image.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Display the image
    ax.imshow(image)
    ax.axis('on')  # Turn on the axis
    plt.show()


# In[49]:


# Check reading image
import matplotlib.image as mpimg
im_path = 'Rosepicture.jpg'
img = mpimg.imread(im_path)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()


# In[50]:


# Call my imshow function
imshow(processed_image)


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[57]:


# Define my predict function
def predict(image_path, model, device, checkpoint_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_to_idx = checkpoint['class_to_idx']

    # Process the image
    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(processed_image)

    # Calculate probabilities and predicted classes
    probabilities = torch.softmax(output, dim=1)
    top_probabilities, top_indices = torch.topk(probabilities, 5)

    # Map indices to classes
    idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices.cpu().numpy()[0]]

    return top_probabilities.cpu().numpy()[0], top_classes


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[58]:


def plot_solution(image_path, model, cat_to_name, device, checkpoint_path):
    # Display the image
    img = PILImage.open(image_path)
    plt.imshow(img)
    plt.axis('off')

    # Make prediction
    top_probabilities, top_classes = predict(image_path, model, device, checkpoint_path)
    class_names = [cat_to_name.get(str(c), "Unknown") for c in top_classes]
    
    # Plot the chart
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 2)
    plt.barh(np.arange(len(top_probabilities)), top_probabilities)
    plt.yticks(np.arange(len(top_probabilities)), class_names)
    plt.xlabel('Probability')
    plt.gca().invert_yaxis()

    plt.show()


# In[59]:


# Try on Rose
image_path = 'Rosepicture.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[60]:


# Try on Japanese Anemone
image_path = 'Japaneseanemone.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[61]:


# Try on Tiger Lily
image_path = 'Firelily.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[62]:


# Try on Orange Dahlia
image_path = 'Orangedahlia.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[63]:


# Try on Camelia
image_path = 'Camelia.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[64]:


# Try on Stemless Gentian
image_path = 'Stemlessgentian.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[66]:


# Try on Monkshood
image_path = 'Monkshood.jpg'
plot_solution(image_path, model, cat_to_name, device, checkpoint_path)


# In[ ]:




