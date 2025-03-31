import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights


# Base directory and subdirectories/files
base_dir = '/jet/home/hlee21/DLNN/dataset/'
train_images_dir = os.path.join(base_dir, 'train_images')
train_csv_path   = os.path.join(base_dir, 'train_data.csv')
test_images_dir  = os.path.join(base_dir, 'test_images')

# Custom Dataset for training data
class MalariaDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.images_dir = os.path.abspath(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_name = self.data_frame.iloc[idx, 0]
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        label = self.data_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom Dataset for test data 
class TestDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = os.path.abspath(images_dir)
        self.transform = transform
        self.image_names = sorted(os.listdir(self.images_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, file_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, file_name

# Define transforms for training and testing
transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = MalariaDataset(csv_file=train_csv_path, images_dir=train_images_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

test_dataset = TestDataset(images_dir=test_images_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

# Build CNN model (1st model)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 56, 56]
        x = x.view(x.size(0), -1)             # flatten
        x = self.fc(x)
        return x

# Resnet18 (2nd model)
if __name__ == '__main__':
    # Set device and initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10  # 10 epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            # accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_dataset)
        accuracy = correct/total
        print(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}, Loss: {epoch_loss:.4f}')

    # Make predictions on the test set
    model.eval()
    predictions = []
    image_ids = []
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            image_ids.extend(names)

    # Save submission file for Kaggle
    submission = pd.DataFrame({'img_name': image_ids, 'label': predictions})
    submission.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")

# Extra credit
import matplotlib.pyplot as plt
import numpy as np
filters = model.conv1.weight.data.clone().cpu()
filters = (filters - filters.min()) / (filters.max() - filters.min())
n_filters = filters.shape[0]
n_cols = 4
n_rows = (n_filters + n_cols - 1) // n_cols
plt.figure(figsize=(n_cols * 2, n_rows * 2))
for i in range(n_filters):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(np.transpose(filters[i].numpy(), (1, 2, 0)))
    plt.axis('off')
plt.suptitle("Conv1 Learned Filters")
plt.savefig('Bonus 1')
plt.show()
