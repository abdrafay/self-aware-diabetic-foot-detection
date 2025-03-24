import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM with model and target layer
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate the Grad-CAM heatmap
        """
        # Forward pass to get logits
        logits = self.model(input_tensor)
        
        # If target class is not specified, use the predicted class
        if target_class is None:
            probs = F.softmax(logits, dim=1)
            _, target_class = torch.max(probs, dim=1)
            target_class = target_class.item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(logits)
        one_hot_output[0, target_class] = 1
        
        # Backward pass
        logits.backward(gradient=one_hot_output, retain_graph=True)
        
        # Compute weights
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activation maps
        for i in range(pooled_gradients.size(0)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Generate heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap, target_class

def apply_heatmap(img, heatmap, alpha=0.5):
    """
    Apply heatmap on image
    """
    # Convert image tensor to numpy array if needed
    if isinstance(img, torch.Tensor):
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        # De-normalize if normalized
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Apply heatmap on image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed_img

def generate_report_with_vlm(image, classification_result, confidence):
    """
    Generate a report using a Vision Language Model
    """
    try:
        # Save the image temporarily
        temp_path = "temp_gradcam.jpg"
        if isinstance(image, np.ndarray):
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image.save(temp_path)
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        model = AutoModelForCausalLM.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        # Prepare image for model
        raw_image = Image.open(temp_path).convert("RGB")
        prompt = "<image>Please provide a medical analysis of this diabetic foot image. The AI model has classified this as " + \
                 f"{classification_result} with {confidence:.2f}% confidence. Focus on visible lesions or abnormalities."
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt")
        
        # Generate report
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=300,
                do_sample=False
            )
            
        # Decode the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Extract the report part after the prompt
        report = generated_text.split("Focus on visible lesions or abnormalities.")[-1].strip()
        return report
    
    except Exception as e:
        return f"Error generating report: {str(e)}. Please check if the required model is available or if there are connection issues."

def xai_module(image_tensor, classifier, classification_result, confidence_score):
    """
    Main explainable AI function that generates heatmaps and reports
    
    Args:
        image_tensor: Input image tensor
        classifier: The classifier model
        classification_result: String representing classification (e.g., "Ulcer" or "Healthy Skin")
        confidence_score: Confidence score (0-100)
    
    Returns:
        dict: Contains visualization and report
    """
    # Determine target layer for Grad-CAM (assuming ResNet backbone's last layer)
    target_layer = classifier.resnet.layer4[-1]
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(classifier, target_layer)
    
    # Generate heatmap
    heatmap, predicted_class = grad_cam.generate_cam(image_tensor)
    
    # Convert image tensor to numpy for visualization
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 0.5 + 0.5)  # De-normalize if your data is normalized to [-1,1]
    img_np = np.uint8(img_np * 255)
    
    # Apply heatmap to original image
    superimposed_img = apply_heatmap(img_np, heatmap)
    
    # Generate medical report using VLM
    report = generate_report_with_vlm(superimposed_img, classification_result, confidence_score * 100)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.text(0.1, 0.5, f"Classification: {classification_result}\n" +
             f"Confidence: {confidence_score*100:.2f}%\n\n" +
             f"Report:\n{report}", 
             fontsize=10, wrap=True)
    plt.axis('off')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig("xai_results.png")
    
    return {
        "visualization": superimposed_img,
        "report": report,
        "classification": classification_result,
        "confidence": confidence_score
    }