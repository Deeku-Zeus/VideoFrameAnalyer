# fashion_transfer_learning.py

import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from PIL import Image
import pandas as pd
import os

class FashionDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(labels_file)  # Ensure CSV has columns: filename, label
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 1]  # Adjust if labels are in a different column
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(image_dir, labels_file, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = FashionDataset(image_dir, labels_file, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader

def build_and_train_model(train_loader, num_classes, num_epochs=5):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), 'deepfashion_model.pth')
    print('Model trained and saved as deepfashion_model.pth')

if __name__ == "__main__":
    # Define paths and parameters
    image_dir = 'path/to/deepfashion/images'
    labels_file = 'path/to/deepfashion/labels.csv'  # CSV with columns: filename, label
    num_classes = 50  # Adjust based on the number of classes in your dataset
    batch_size = 32
    num_epochs = 5

    # Load data
    train_loader = load_data(image_dir, labels_file, batch_size)

    # Build and train model
    build_and_train_model(train_loader, num_classes, num_epochs)
