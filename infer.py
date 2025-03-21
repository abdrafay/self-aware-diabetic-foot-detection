import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torch
import torchvision.transforms as transforms
from PIL import Image
from model.vae import VAE
import torch.nn.functional as F
from model.classifier import VAEClassifier
from anomaly_detection import anomaly_detection

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
    """
    Runs an image through the VAE and classifier.
    Returns:
        original image tensor, reconstructed tensor, and raw classifier logits.
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get VAE outputs
        reconstructed, mu, logvar = vae(image_tensor)
        # Get classifier logits
        logits = classifier(image_tensor)

    return image_tensor, reconstructed, logits

# Example usage
image_path = "dataset/img3-arch-support.jpg"
original, reconstructed, logits = infer(image_path)

# Run anomaly detection
decision, calibrated_probs = anomaly_detection(logits, original, reconstructed)

# Display final results
formatted_probs = [f"{p*100:.2f}%" for p in calibrated_probs[0]]
print("\nFinal Decision:", decision)
# print("Calibrated Confidence Scores:", formatted_probs)

# save the reconstructed image in reconsructed/ folder
vutils.save_image(reconstructed, "reconstructed/" + image_path.split("/")[-1], normalize=True)
