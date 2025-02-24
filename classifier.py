import torch.nn.functional as Fdaa
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.models as models

class VAEClassifier(nn.Module):
    def __init__(self, vae, latent_dim=256, num_classes=2):
        super(VAEClassifier, self).__init__()
        self.vae_encoder = vae.encoder  # Use the encoder from the VAE
        self.vae_latent = nn.Sequential(
            nn.Linear(512 * 4 * 4, latent_dim),  # Match the VAE latent space
            nn.ReLU()
        )

        # Load a pre-trained ResNet backbone and modify it
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # Additional CNN layers after ResNet
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + 128, 64),  # Combine VAE and ResNet features
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Extract features from the VAE encoder
        vae_features = self.vae_encoder(x)
        vae_features = vae_features.view(vae_features.size(0), -1)
        vae_latent = self.vae_latent(vae_features)

        # Pass input through ResNet backbone
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), 512, 1, 1)
        resnet_features = self.cnn_layers(resnet_features)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)

        # Concatenate VAE and ResNet features
        combined_features = torch.cat([vae_latent, resnet_features], dim=1)

        # Classification
        output = self.classifier(combined_features)
        return output
    
