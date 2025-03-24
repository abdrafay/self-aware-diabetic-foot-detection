import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.transforms as transforms
from PIL import Image
from model.vae import VAE
import torch.nn.functional as F
from model.classifier import VAEClassifier
from self_aware_module import self_awareness
from explainable_ai import xai_module
import matplotlib.pyplot as plt
import cv2

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
        reconstructed, _, _ = vae(image_tensor)
        # Get classifier logits
        logits = classifier(image_tensor)

    return image_tensor, reconstructed, logits

# Example usage
image_path = "dataset/img3-arch-support.jpg"
original, reconstructed, logits = infer(image_path)

# Run anomaly detection
text, decision = self_awareness(logits, original, reconstructed)
print("\nSelf-Aware Output:", text)

# if decision:
#     # Calculate probabilities from logits
#     probs = F.softmax(logits, dim=1)
#     confidence, predicted = torch.max(probs, 1)
#     predicted = predicted.item()
#     confidence = confidence.item()
#     classification_result = 'Ulcer' if predicted == 0 else 'Healthy Skin'
    
#     print(f"Final Decision: {classification_result} (Confidence: {confidence*100:.2f}%)")
    
#     # Generate explainable AI results
#     xai_results = xai_module(original, classifier, classification_result, confidence)
    
#     # Display results
#     print("\nXAI Report:")
#     print("-" * 50)
#     print(xai_results["report"])
#     print("-" * 50)
#     print("Visualization saved as 'xai_results.png'")
# else:
#     print("The model is uncertain about this image. Please consult a healthcare professional.")