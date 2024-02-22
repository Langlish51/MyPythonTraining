import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
import json
from PIL import Image
import argparse

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_layers = checkpoint['hidden_layers']
    output_size = checkpoint['output_size']
    drop_p = checkpoint['drop_p']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = CustomClassifier(model.classifier[0].in_features, output_size, hidden_layers, drop_p)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = CustomClassifier(model.fc.in_features, output_size, hidden_layers, drop_p)
    else:
        raise ValueError("Unsupported architecture")

    model.load_state_dict(state_dict)
    model.class_to_idx = class_to_idx

    return model

class CustomClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = nn.functional.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)
        return nn.functional.log_softmax(x, dim=1)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = img_transforms(image)
    return image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    img = Image.open(image_path)
    img = process_image(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs, indices = torch.topk(torch.exp(output), topk)
        probs = probs.numpy().squeeze()
        indices = indices.numpy().squeeze()

    return probs, indices

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    probs, indices = predict(args.image_path, model, args.top_k)

    class_names = []
    for idx in indices:
        try:
            class_names.append(cat_to_name[str(idx)])
        except KeyError:
            print(f"KeyError: Index {idx} not found in cat_to_name dictionary")

    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
        print(f"Rank {i+1}: {class_name} with probability {prob:.3f}")

if __name__ == "__main__":
    main()
