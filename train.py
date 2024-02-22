import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import argparse
import json
import os
import torch.nn.functional as F

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the parser
parser = argparse.ArgumentParser(description='Train an image classifier on a dataset')
parser.add_argument('data_directory', type=str, help='Indicate dataset directory')
parser.add_argument('--save_dir', type=str, default='checkpoints', help='Checkpoint directory')
parser.add_argument('--arch', type=str, choices=['vgg16', 'resnet18'], default='vgg16', help='2 models architectures')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_layers', type=int, nargs='+', default=[512], help='Number of units in hidden layers')
parser.add_argument('--output_size', type=int, default=102, help='Size of the output layer')
parser.add_argument('--drop_p', type=float, default=0.5, help='Dropout probability')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()

def initialize_model(arch, num_labels, hidden_units, drop_p):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[0].in_features
        model.classifier = CustomClassifier(num_features, num_labels, hidden_units, drop_p)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = CustomClassifier(num_features, num_labels, hidden_units, drop_p)
    else:
        raise ValueError("Invalid model name, VGG or Resnet only")
    return model

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

# Data loading
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = args.data_directory
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid']}
dataloaders['test'] = DataLoader(image_datasets['test'], batch_size=32, shuffle=False)

# Load label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Initialize the model
model = initialize_model(args.arch, args.output_size, args.hidden_layers, args.drop_p).to(device)

# Define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters() if args.arch == 'vgg16' else model.fc.parameters(), lr=args.learning_rate)

# Train network
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation pass
    validation_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)
            validation_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
          f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
          f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

# Testing my network
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model.forward(inputs)
        batch_loss = criterion(outputs, labels)
        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")

# Save the checkpoint
checkpoint = {
    'arch': args.arch,
    'state_dict': model.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx,
    'hidden_layers': args.hidden_layers,
    'drop_p': args.drop_p,
    'output_size': args.output_size,
    'epochs': args.epochs,
    'optimizer_state_dict': optimizer.state_dict(),
}

checkpoint_path = 'checkpoint.pth'
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")


