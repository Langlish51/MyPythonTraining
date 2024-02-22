# AI Programming with Python Project

**PROJECT OVERVIEW:**
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

**CREATOR**
Franck Tujek - 22/02/2024

**FILES:**
- The "FTujek Image Classifier Project" file contains the Jupyter Notebook for the design, training, testing and evaluation of my model using VGG16
- The "train.py" file is a Python script designed to train a flower classification model. It allows users to choose between VGG16 and ResNet18, for training the model. Other optional arguments are learning rate, hidden layers, output size, dropout probability, number of epochs, and GPU usage.
- The "predict.py" is a Python script used for making predictions with a pre-trained flower classification model. It takes an input image and a pre-trained model checkpoint as arguments and returns the top K most likely classes along with their probabilities. Users can customize the prediction process using image path, checkpoint, top K number and different mapping for categories to name
