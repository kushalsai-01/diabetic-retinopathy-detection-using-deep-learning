import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        score = output[0, target_class]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam

    def overlay(self, heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        heatmap_uint8 = np.uint8(255 * heatmap)
        colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        h, w = original_image.shape[:2]
        colormap = cv2.resize(colormap, (w, h))
        
        blended = cv2.addWeighted(colormap, alpha, original_image, 1 - alpha, 0)
        return blended
