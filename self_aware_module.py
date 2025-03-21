import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import time
import subprocess
import logging
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# Constants & Hyperparameters
# -----------------------------
BASE_RECON_THRESHOLD = 0.1    # Base threshold for reconstruction error
MAX_RECON_ERROR    = 0.6      # If reconstruction error exceeds this, discard input
ENTROPY_THRESHOLD  = 0.4      # Entropy value above which the classifier is considered uncertain
TEMPERATURE        = 2.0      # Temperature for scaling logits (T > 1 softens the probabilities)
MC_DROPOUT_ITER    = 10       # Number of iterations for MC Dropout

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
writer = SummaryWriter()

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

def mc_dropout(model, inputs, iterations=MC_DROPOUT_ITER):
    """
    Perform Monte Carlo Dropout to estimate uncertainty.
    """
    model.train()  # Enable dropout
    probs = []
    with torch.no_grad():
        for _ in range(iterations):
            outputs = model(inputs)
            probs.append(torch.softmax(outputs, dim=1))
    probs = torch.stack(probs)
    mean_probs = probs.mean(dim=0)
    return mean_probs

def is_uncertain(probs):
    """
    Determine if the classifier is uncertain based on entropy.
    """
    entropy = compute_entropy(probs)
    logger.info(f"Computed entropy: {entropy:.4f}")
    return entropy > ENTROPY_THRESHOLD


def save_image(image, folder, save_id):
    """
    Save the image to the specified folder with the given ID.
    """
    image_save_path = os.path.join(folder, f"{save_id}.jpg")
    vutils.save_image(image, image_save_path, normalize=True)
    logger.info(f"Image saved in {folder}/{save_id}.jpg folder.")

def trigger_training():
    """
    Trigger the training process by calling train.py with specific arguments.
    """
    try:
        subprocess.run(
            [
                "python", "train.py",
                "--vae_epochs", "1",
                "--classifier_epoch", "1",
                "--batch_size", "128"
            ],
            check=True
        )
        logger.info("Training triggered successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Error during training trigger:", e)

# -----------------------------
# Incremental Learning Logic
# -----------------------------
def incremental_learning(image, confidence, is_ood):
    """
    Incremental learning logic to handle the decision-making process.
    1. Check classifier uncertainty (entropy)
    2. Check if image is an anomaly (VAE reconstruction error)
    3. Decide next action

    Parameters:
        image_path : Path to the input image.
        recon_error: Reconstruction error from the VAE.
        confidence : Classifier confidence scores.
        is_ood     : Boolean indicating if the sample is out-of-distribution.

    Returns:
        None
    """
    # TODO: Implement the logic to first take input from expert
    
    abnormal_prob = confidence[0][0].item()  # probability of being abnormal
    normal_prob = confidence[0][1].item()    # probability of being normal

    logger.info(f"Abnormal probability: {abnormal_prob:.4f}")
    logger.info(f"Normal probability: {normal_prob:.4f}")

    if abnormal_prob > normal_prob:
        folder = "dataset/Abnormal(Ulcer)"
    else:
        folder = "dataset/Normal(Healthy skin)"
    
    save_id = int(time.time() * 1000)
    save_image(image, folder, save_id)

    if is_ood:
        logger.info("Out-of-distribution sample detected. Triggering retraining.")
        trigger_training()
    else:
        logger.info("Misclassified known class.")

# -----------------------------
# Self Awareness Module
# -----------------------------
def self_awareness(model, logits, original, reconstructed, base_threshold=BASE_RECON_THRESHOLD, max_threshold=MAX_RECON_ERROR, 
temperature=TEMPERATURE):
    """
    Combines classifier calibration, uncertainty estimation, and VAE reconstruction error.
    
    Parameters:
        model          : Classifier model with dropout layers.
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
    
    # Perform MC Dropout to estimate uncertainty
    mc_probs = mc_dropout(model, original)
    
    # Anomaly detection by checking uncertainty
    uncertain = is_uncertain(mc_probs)
    
    # Compute the VAE reconstruction error
    recon_error = compute_reconstruction_error(original, reconstructed)
    
    # Compute statistics for adaptive thresholding
    recon_errors = [recon_error]  # This should be a list of past errors
    mean_error = np.mean(recon_errors)
    std_error = np.std(recon_errors)
    adaptive_threshold = mean_error + std_error
    logger.info(f"Adaptive threshold: {adaptive_threshold:.4f}")
    logger.info(f"Reconstruction error: {recon_error:.4f}")
    
    # Decision logic based on reconstruction error and uncertainty
    if recon_error >= max_threshold:
        return "❌ High reconstruction error (likely OOD). Discarding input.", False
    elif adaptive_threshold < recon_error < max_threshold or uncertain:
        is_ood = recon_error >= max_threshold
        incremental_learning(original, mc_probs, is_ood)
        return "⚠️ Elevated reconstruction error or high uncertainty. Review needed.", False
    else:
        return "✅ Prediction accepted based on low reconstruction error and low uncertainty.", True

