from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_layer_name: Optional[str] = None,
    ):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        if target_layer is not None:
            self.target_layer = target_layer
        elif target_layer_name is not None:
            self.target_layer = self._find_layer_by_name(target_layer_name)
        else:
            self.target_layer = self._find_last_conv_layer()
            
        self._register_hooks()
        
    def _find_layer_by_name(self, name: str) -> nn.Module:
        for n, module in self.model.named_modules():
            if n == name:
                return module
        raise ValueError(f"Layer {name} not found in model")
        
    def _find_last_conv_layer(self) -> nn.Module:
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No convolutional layer found in model")
        return last_conv
        
    def _register_hooks(self) -> None:
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
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam.cpu().numpy()
    
    def generate_cam_batch(
        self,
        input_tensors: torch.Tensor,
        target_classes: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        cams = []
        for i in range(input_tensors.size(0)):
            target = target_classes[i] if target_classes else None
            cam = self.generate_cam(input_tensors[i:i+1], target_class=target)
            cams.append(cam)
        return cams


class GradCAMPlusPlus(GradCAM):
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
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
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]),
            Image.BILINEAR
        )
    ) / 255.0
    
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    
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
    gradcam = GradCAM(model, target_layer=target_layer)
    
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
    cam = gradcam.generate_cam(image, target_class=target_class or pred_class)
    overlay = overlay_cam_on_image(original_image, cam)
    
    if show or save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
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
