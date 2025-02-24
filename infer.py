# warning remove code
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torch
import torchvision.transforms as transforms
from PIL import Image
from vae import VAE
import torch.nn.functional as F
from classifier import VAEClassifier
import numpy as np

# Load trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize VAE and Classifier
vae = VAE(img_channels=3, latent_dim=256).to(device)
classifier = VAEClassifier(vae, latent_dim=256, num_classes=2).to(device)

# Load model weights
vae.load_state_dict(torch.load("models/vae_epoch_100.pth", map_location=device))
classifier.load_state_dict(torch.load("models/classifier.pth", map_location=device))

# Set models to evaluation mode
vae.eval()
classifier.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match VAE input
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1,1] to match Tanh output
])

def infer(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Pass through VAE
    with torch.no_grad():
        reconstructed, mu, logvar = vae(image)  # Reconstruct image and get latent features
        latent_features = mu  # Use mean (mu) as the latent representation

        # Classify using VAE features
        logits = classifier(image)  # Get class scores
        # probabilities = torch.sigmoid(logits)  # Convert to probabilities
        probabilities = torch.softmax(logits, dim=1)  # Already applied
        formatted_probs = [f"{p*100:.2f}%" for p in probabilities[0]]
        print("Confidence Scores:", formatted_probs)

    # Return reconstructed image, latent features, and classification probabilities
    return reconstructed.cpu(), latent_features.cpu(), probabilities.cpu()


def compute_reconstruction_error(original, reconstructed):
    """
    Measures the difference between original and VAE-reconstructed image.
    If error is high, image is an anomaly.
    """
    # Ensure both tensors are on the same device
    device = original.device  # Get the device of the original tensor
    reconstructed = reconstructed.to(device)  # Move reconstructed to the same device

    return F.mse_loss(original, reconstructed).item()

def self_aware_module(probs, original, reconstructed, recon_threshold=0.1):
    """
    Main self-awareness logic:
    1. Check classifier uncertainty (entropy)
    2. Check if image is an anomaly (VAE reconstruction error)
    3. Decide next action
    """
    uncertainty = is_uncertain(probs)
    recon_error = compute_reconstruction_error(original, reconstructed)

    # Dynamically adjust reconstruction threshold
    adaptive_threshold = recon_threshold * (1 + uncertainty * 0.5)  # Increase if uncertain

    if uncertainty or recon_error > adaptive_threshold:
        print("⚠️ Uncertain or Anomalous! Sending for expert/self-learning.")
        return "Review Needed"
    else:
        print("✅ Confident Prediction! Using result.")
        return "Prediction Accepted"


THRESHOLD = 0.4  # Adjust based on experiments

def compute_entropy(probs):
    """
    Compute entropy of probability distribution.
    Avoids log(0) issues by using np.nan_to_num.
    """
    probs = probs.detach().cpu().numpy()  # Convert tensor to numpy
    probs = np.clip(probs, 1e-10, 1.0)  # Ensure no zero values
    entropy = -np.sum(probs * np.log2(probs))  # Compute entropy
    return entropy


def is_uncertain(probs):
    """
    Checks if the prediction is uncertain.
    Returns: True if uncertain, False otherwise
    """
    return compute_entropy(probs) > THRESHOLD


# Example usage
image_path = "Abga_standee.png"
reconstructed_image, vae_features, confidence = infer(image_path)

# find mse of reconstructed image
original = Image.open(image_path).convert("RGB")
original = transform(original).unsqueeze(0).to(device)
recon_error = compute_reconstruction_error(original, reconstructed_image)


# save the reconstructed image in reconsructed/ folder
vutils.save_image(reconstructed_image, "reconstructed/5.jpg", normalize=True)

# Print results
print("Reconstruction Error:", recon_error)
print("Is the model uncertain?", is_uncertain(confidence))

# Example usage
self_aware_module(probs=confidence, original=original, reconstructed=reconstructed_image, recon_threshold=THRESHOLD)
