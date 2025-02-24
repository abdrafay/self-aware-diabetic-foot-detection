import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VAEClassifier(nn.Module):
    def __init__(self, vae, latent_dim=256, num_classes=2, dropout_prob=0.5):
        super(VAEClassifier, self).__init__()
        # Use the encoder from the VAE
        self.vae_encoder = vae.encoder  
        self.vae_latent = nn.Sequential(
            nn.Linear(512 * 4 * 4, latent_dim),  # Adjust to match your VAE encoder output shape
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        # Load a pre-trained ResNet backbone and modify it
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer

        # Additional CNN layers after ResNet with dropout
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Fully connected layers for classification with dropout
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + 128, 64),  # Combine VAE and ResNet features
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # Extract features from the VAE encoder
        vae_features = self.vae_encoder(x)
        vae_features = vae_features.view(vae_features.size(0), -1)
        vae_latent = self.vae_latent(vae_features)

        # Extract features using the ResNet backbone
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), 512, 1, 1)
        resnet_features = self.cnn_layers(resnet_features)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)

        # Concatenate VAE and ResNet features
        combined_features = torch.cat([vae_latent, resnet_features], dim=1)

        # Classification
        output = self.classifier(combined_features)
        return output
