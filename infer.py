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
from anomaly_detection import compute_reconstruction_error, is_uncertain, anomaly_detection, THRESHOLD

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
anomaly_detection(probs=confidence, original=original, reconstructed=reconstructed_image, recon_threshold=THRESHOLD)
