# Removing the warnings
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from model.vae import VAE, vae_loss
from model.classifier import VAEClassifier

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE and its Classifier")
    parser.add_argument('--vae_epochs', type=int, default=100, 
                        help='Number of epochs to train the VAE')
    parser.add_argument('--classifier_epochs', type=int, default=20, 
                        help='Number of epochs to train the classifier')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=10, 
                        help='Interval of epochs to save the VAE model')
    # Add additional arguments if needed
    return parser.parse_args()

def get_dataloaders(batch_size, image_size=128):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    dataset = datasets.ImageFolder(root="dataset/", transform=transform)
    
    # Split Calculation
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Ensure no data is left out
    
    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Train method for VAE
def train_vae(model, loader, epochs, save_interval, save_path="vae_epoch_{}.pth"):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nVAE Epoch [{epoch+1}/{epochs}]")
        for images, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
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
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('models', model_filename))
            print(f"Model saved at {model_filename}")

def train_classifier(model, train_loader, val_loader, epochs, criterion, optimizer):
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        print(f"\nClassifier Epoch [{epoch+1}/{epochs}]")
        
        model.train()
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
    
    return train_loss_list, train_acc_list, val_loss_list, val_acc_list

def save_classifier(model, path):
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), os.path.join('models', path))
    print(f"Classifier saved at {os.path.join('models', path)}")

def main():
    args = parse_args()
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.batch_size)
    
    # Initialize or train VAE model
    vae = VAE(img_channels=3, latent_dim=256).to(device)
    
    vae_model_path = os.path.join('models', "vae_epoch_100.pth")
    if os.path.exists(vae_model_path):
        vae.load_state_dict(torch.load(vae_model_path, map_location=device))
        print("Loaded pretrained VAE model.")
    else:
        print("Pretrained VAE model not found. Please train the VAE first.")
        return
    
    if args.vae_epochs == 0:
        # print("VAE training skipped as epochs=0.")
        pass
    else: 
        # Train the VAE
        train_vae(vae, train_loader, epochs=args.vae_epochs, save_interval=args.save_interval)


    # Optionally, extract latent representations if needed later.
    # For classifier training, we use the VAE's encoder inside VAEClassifier.
    
    # Instantiate the classifier
    classifier = VAEClassifier(vae, latent_dim=256, num_classes=2).to(device)
    
    # Define loss function and optimizer for the classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    classifier_model_path = os.path.join('models', "classifier.pth")

    # load the classifier model
    
    if os.path.exists(classifier_model_path):
        classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))

    # check if epochs is not 0
    if args.classifier_epochs == 0:
        # print("Classifier training skipped as epochs=0.")
        pass
    else:
        # Train the classifier
        train_classifier(classifier, train_loader, val_loader, epochs=args.classifier_epochs, criterion=criterion, optimizer=optimizer)
    
    # Save the classifier
    save_classifier(classifier, "classifier.pth")

if __name__ == "__main__":
    print("\n---------------------------------------------------------------------\n")
    print("Device:", device)
    print("Training VAE and Classifier...")
    main()
