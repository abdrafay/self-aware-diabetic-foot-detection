import numpy as np
import torch.nn.functional as F

def compute_reconstruction_error(original, reconstructed):
    """
    Measures the difference between original and VAE-reconstructed image.
    If error is high, image is an anomaly.
    """
    # Ensure both tensors are on the same device
    device = original.device  # Get the device of the original tensor
    reconstructed = reconstructed.to(device)  # Move reconstructed to the same device

    return F.mse_loss(original, reconstructed).item()

def anomaly_detection(probs, original, reconstructed, recon_threshold=0.1):
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

