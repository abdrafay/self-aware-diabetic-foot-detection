import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import time
import subprocess

# -----------------------------
# Constants & Hyperparameters
# -----------------------------
BASE_RECON_THRESHOLD = 0.1    # Base threshold for reconstruction error
MAX_RECON_ERROR    = 0.6      # If reconstruction error exceeds this, discard input
ENTROPY_THRESHOLD  = 0.4      # Entropy value above which the classifier is considered uncertain
TEMPERATURE        = 2.0      # Temperature for scaling logits (T > 1 softens the probabilities)

# -----------------------------
# Utility Functions
# -----------------------------
def compute_reconstruction_error(original, reconstructed):
    """
    Compute Mean Squared Error between original and reconstructed images.
    """
    device = original.device  # Ensure both tensors are on the same device
    reconstructed = reconstructed.to(device)
    return F.mse_loss(original, reconstructed).item()

def compute_entropy(probs):
    """
    Compute the entropy of the probability distribution.
    Clipping avoids issues with log(0).
    """
    probs_np = probs.detach().cpu().numpy()
    probs_np = np.clip(probs_np, 1e-10, 1.0)
    entropy = -np.sum(probs_np * np.log2(probs_np))
    return entropy

def is_uncertain(probs):
    """
    Determine if the classifier is uncertain based on entropy.
    """
    entropy = compute_entropy(probs)
    print(f"Computed entropy: {entropy:.4f}")
    return entropy > ENTROPY_THRESHOLD

def calibrated_softmax(logits, temperature=TEMPERATURE):
    """
    Apply temperature scaling to the logits and then compute softmax.
    This calibration helps to soften overconfident predictions.
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=1)


# -----------------------------
# Self Awareness Module
# -----------------------------
def self_awareness_module(image, recon_error, confidence):
    """
    Main self-awareness logic:
    1. Check classifier uncertainty (entropy)
    2. Check if image is an anomaly (VAE reconstruction error)
    3. Decide next action

    Parameters:
        image_path : Path to the input image.
        recon_error: Reconstruction error from the VAE.
        confidence : Classifier confidence scores.

    Returns:
        None
    """
    abnormal_prob = confidence[0][0].item()  # probability of being abnormal
    normal_prob = confidence[0][1].item()    # probability of being normal

    print(f"Abnormal probability: {abnormal_prob:.4f}")
    print(f"Normal probability: {normal_prob:.4f}")

    if abnormal_prob > normal_prob:
        folder = "dataset/Abnormal(Ulcer)"
    else:
        folder = "dataset/Normal(Healthy skin)"
    
    save_id = int(time.time() * 1000)
    image_save_path = os.path.join(folder, f"{save_id}.jpg")

    vutils.save_image(image, image_save_path, normalize=True)
    
    print(f"Image saved in {folder}/{save_id}.jpg folder.")

    try:
        # Pass additional command-line arguments to train.py
        subprocess.run(
            [
                "python", "train.py",
                "--vae_epochs", "1",
                "--classifier_epoch", "1",
                "--batch_size", "128"
            ],
            check=True
        )
        print("Training triggered successfully.")
    except subprocess.CalledProcessError as e:
        print("Error during training trigger:", e)
    

# -----------------------------
# Main Anomaly Detection Logic
# -----------------------------
def anomaly_detection(logits, original, reconstructed, base_threshold=BASE_RECON_THRESHOLD, max_threshold=MAX_RECON_ERROR, 
temperature=TEMPERATURE):
    """
    Combines classifier calibration, uncertainty estimation, and VAE reconstruction error.
    
    Parameters:
        logits         : Raw output from the classifier.
        original       : Original input image tensor.
        reconstructed  : Reconstructed image tensor from the VAE.
        base_threshold : Base reconstruction error threshold.
        max_threshold  : Maximum reconstruction error allowed (beyond which input is discarded).
        temperature    : Temperature for scaling classifier logits.
    
    Returns:
        decision       : One of "Discard", "Review Needed", or "Prediction Accepted".
        probs          : Calibrated probabilities after temperature scaling.
    """
    # Calibrate the classifier predictions
    probs = calibrated_softmax(logits, temperature)
    
    # Check for classifier uncertainty (via entropy)
    uncertain = is_uncertain(probs)
    
    # Compute the VAE reconstruction error
    recon_error = compute_reconstruction_error(original, reconstructed)
    
    # Adjust threshold based on classifier uncertainty
    adaptive_threshold = base_threshold * (1 + (0.5 if uncertain else 0))
    print(f"Adaptive threshold: {adaptive_threshold:.4f}")
    print(f"Reconstruction error: {recon_error:.4f}")
    
    # Decision logic based on reconstruction error
    if recon_error >= max_threshold:
        print("❌ High reconstruction error (likely OOD). Discarding input.")
        return "Discard", probs
    elif adaptive_threshold < recon_error < max_threshold:
        print("⚠️ Elevated reconstruction error. Routing to further self-awareness module.")
        self_awareness_module(original, recon_error, probs)
        return "Review Needed", probs
    else:
        print("✅ Prediction accepted based on low reconstruction error.")
        return "Prediction Accepted", probs
    
