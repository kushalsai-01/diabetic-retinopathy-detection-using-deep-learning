"""
Grad-CAM implementation for model interpretability.
Visualizes which regions of the retinal image influence classification.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Generates visual explanations showing which image regions
    most influenced the model's prediction. Critical for clinical
    validation and building trust in AI-assisted diagnosis.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization", ICCV 2017
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_layer_name: Optional[str] = None,
    ):
        """
        Args:
            model: Trained classification model.
            target_layer: Target convolutional layer for CAM.
            target_layer_name: Name of target layer (alternative to target_layer).
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        
        # Find target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif target_layer_name is not None:
            self.target_layer = self._find_layer_by_name(target_layer_name)
        else:
            self.target_layer = self._find_last_conv_layer()
            
        # Register hooks
        self._register_hooks()
        
    def _find_layer_by_name(self, name: str) -> nn.Module:
        """Find layer by name."""
        for n, module in self.model.named_modules():
            if n == name:
                return module
        raise ValueError(f"Layer {name} not found in model")
        
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No convolutional layer found in model")
        return last_conv
        
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W).
            target_class: Target class for CAM. If None, uses predicted class.
            
        Returns:
            CAM heatmap of shape (H, W) with values in [0, 1].
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam.cpu().numpy()
    
    def generate_cam_batch(
        self,
        input_tensors: torch.Tensor,
        target_classes: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Generate CAM for a batch of images."""
        cams = []
        for i in range(input_tensors.size(0)):
            target = target_classes[i] if target_classes else None
            cam = self.generate_cam(
                input_tensors[i:i+1],
                target_class=target,
            )
            cams.append(cam)
        return cams


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ with improved weighting scheme.
    
    Uses pixel-wise weighting of gradients for better localization
    of multiple instances of the same class.
    
    Reference: Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based
    Visual Explanations for Deep Convolutional Networks", WACV 2018
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM++ weighting
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Second and third order gradients approximation
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Alpha computation
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        
        # Apply ReLU to gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam.cpu().numpy()


def overlay_cam_on_image(
    image: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, 3) with values in [0, 255].
        cam: CAM heatmap (H, W) with values in [0, 1].
        alpha: Transparency of heatmap overlay.
        colormap: Matplotlib colormap name.
        
    Returns:
        Overlaid image as numpy array.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    # Resize CAM to match image size
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR
        )
    ) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Overlay
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    
    return overlay


def visualize_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    original_image: Union[np.ndarray, Image.Image],
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Generate and visualize Grad-CAM for an image.
    
    Args:
        model: Trained model.
        image: Preprocessed input tensor (1, C, H, W).
        original_image: Original image for overlay.
        target_class: Target class (None for predicted class).
        target_layer: Target layer for CAM.
        save_path: Path to save visualization.
        show: Whether to display the visualization.
        
    Returns:
        Tuple of (overlay_image, predicted_class).
    """
    gradcam = GradCAM(model, target_layer=target_layer)
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
    # Generate CAM
    cam = gradcam.generate_cam(image, target_class=target_class or pred_class)
    
    # Create overlay
    overlay = overlay_cam_on_image(original_image, cam)
    
    if show or save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # CAM heatmap
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: Class {pred_class} ({confidence:.2%})")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            
        if show:
            plt.show()
        else:
            plt.close()
            
    return overlay, pred_class
