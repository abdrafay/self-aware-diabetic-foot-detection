import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split
import torchvision.models as models  
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from vae import VAE, vae_loss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from classifier import VAEClassifier
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading & PreProcessing Data

image_size = 128  # You can increase this if needed

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

dataset = datasets.ImageFolder(root="dataset/", transform=transform)

train_split = 0.7
val_split = 0.15
test_split = 0.15

# Split Calculation
total_size = len(dataset)
train_size = int(train_split * total_size)
val_size = int(val_split * total_size)
test_size = total_size - train_size - val_size  # Ensure no data is left out

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Dataloaders
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train Method ( VAE )
def train_vae(model, loader, epochs=20, save_interval=10, save_path="vae_epoch_{}.pth"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        for images, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)  # Move images to GPU
            optimizer.zero_grad()
            x_recon, mu, logvar = model(images)
            loss = vae_loss(images, x_recon, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        
        # Save model every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            model_filename = save_path.format(epoch + 1)
            torch.save(model.state_dict(), os.path.join('models',model_filename))
            print(f"Model saved at {model_filename}")

vae = VAE(img_channels=3, latent_dim=256).to(device)
# train_vae(vae, train_loader, epochs=100, save_interval=10)
# load vae model
vae.load_state_dict(torch.load("vae_epoch_100.pth", map_location=device))

# Step 1: Extract latent representations
vae.eval()  # Set the model to evaluation mode
latent_vectors = []
labels = []

with torch.no_grad():
    for images, label in tqdm(test_loader, desc="Extracting latent vectors"):
        images = images.to(device)
        mu, logvar = vae.encode(images)
        z = vae.reparameterize(mu, logvar)  # Use the latent vector
        latent_vectors.append(z.cpu().numpy())
        labels.extend(label.numpy())  # Assuming your dataset has labels

latent_vectors = np.concatenate(latent_vectors)
labels = np.array(labels)

# Instantiate the model
classifier = VAEClassifier(vae, latent_dim=256, num_classes=2).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

# Training function for classifier
def train_classifier(model, train_loader, val_loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100 * correct / total
        train_acc_list.append(accuracy)
        train_loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Validation step
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_avg_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_avg_loss)
        print(f"Validation Loss: {val_avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

def save_classifier(model, path):
    """
    Save the classifier model to the specified path
    Args:
        model: The VAEClassifier model to save
        path: Path where the model should be saved
    """
    torch.save(model.state_dict(), os.path.join('models',path))

# Train the classifier
train_classifier(classifier, train_loader, val_loader, epochs=20)

# Save the classifier
save_classifier(classifier, "classifier.pth")