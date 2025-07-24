"""
ComfyUI DAViD Multitask Node
Performs depth estimation, surface normal estimation, and foreground segmentation in a single pass
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import folder_paths

from ..runtime.multi_task_estimator import MultiTaskEstimator
import comfy.model_management

class DAViDMultiTaskNode:
    """
    DAViD Multi-Task Node - Performs depth, normal, and foreground estimation in one pass
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Check for available models
        model_dir = os.path.join(folder_paths.models_dir, "david")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # List available models
        available_models = []
        for file in os.listdir(model_dir) if os.path.exists(model_dir) else []:
            if file.endswith('.onnx'):
                available_models.append(file)
        
        # Default models if none found
        if not available_models:
            available_models = ["multitask-vitl16_384.onnx"]
        
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "model_name": (available_models, {"default": available_models[0]}),
                "inverse_depth": ("BOOLEAN", {"default": False}),
                "binarize_foreground": ("BOOLEAN", {"default": False}),
                "foreground_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("depth_map", "normal_map", "foreground_rgb", "foreground_mask")
    FUNCTION = "process"
    CATEGORY = "DAViD"
    
    def __init__(self):
        self.model = None
        self.current_model_name = None
        
    def load_model(self, model_name):
        """Load the ONNX model if not already loaded"""
        if self.model is None or self.current_model_name != model_name:
            # Try multiple possible model locations
            possible_paths = [
                os.path.join(folder_paths.models_dir, "david", model_name),
                os.path.join(os.path.dirname(__file__), "..", "models", "david", model_name),
                os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-DAViD", "models", "david", model_name),
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"Model '{model_name}' not found in any of these locations:\n" +
                    "\n".join(f"  - {p}" for p in possible_paths) +
                    f"\n\nPlease download the multitask model from:\n"
                    f"https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx\n"
                    f"and save it to: {possible_paths[0]}"
                )
            
            # Initialize the model
            self.model = MultiTaskEstimator(
                onnx_model=model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                is_inverse_depth=False  # We'll handle this in post-processing
            )
            self.current_model_name = model_name
            print(f"Loaded DAViD model from: {model_path}")
            
    def process(self, image, model_name="multitask-vitl16_384.onnx", 
                inverse_depth=False, binarize_foreground=False, foreground_threshold=0.5):
        """
        Process the image through the DAViD multitask model
        
        Args:
            image: ComfyUI image tensor (B, H, W, C) in RGB format with values in [0, 1]
            model_name: Name of the model file
            inverse_depth: Whether to invert the depth map
            binarize_foreground: Whether to binarize the foreground mask
            foreground_threshold: Threshold for binarization
        """
        # Load model if needed
        self.load_model(model_name)
        
        # Get the device
        device = comfy.model_management.get_torch_device()
        
        # Process each image in the batch
        batch_size = image.shape[0]
        depth_results = []
        normal_results = []
        foreground_rgb_results = []
        foreground_mask_results = []
        
        for i in range(batch_size):
            # Convert from ComfyUI format (RGB, float32, 0-1) to OpenCV format (BGR, uint8, 0-255)
            img_rgb = image[i].cpu().numpy()
            img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = self.model.estimate_all_tasks(img_bgr)
            
            # Process depth map
            depth_map = results["depth"]
            if inverse_depth:
                depth_map = -depth_map
            
            # Normalize depth to [0, 1] for visualization
            depth_min = np.min(depth_map)
            depth_max = np.max(depth_map)
            if depth_max > depth_min:
                depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth_map)
            
            # Convert depth to RGB for visualization (using a colormap)
            depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            # Process normal map
            normal_map = results["normal"]
            # Convert from [-1, 1] to [0, 1] for visualization
            normal_vis = (normal_map + 1.0) / 2.0
            normal_vis = np.clip(normal_vis, 0, 1)
            
            # Process foreground mask
            foreground_mask = results["foreground"]
            
            # Ensure mask is 2D
            if len(foreground_mask.shape) == 3:
                foreground_mask = foreground_mask[:, :, 0]
            
            # Binarize if requested
            if binarize_foreground:
                foreground_mask = (foreground_mask > foreground_threshold).astype(np.float32)
            
            # Create RGB visualization of foreground
            foreground_rgb = np.stack([foreground_mask] * 3, axis=-1)
            
            # Append results
            depth_results.append(torch.from_numpy(depth_rgb))
            normal_results.append(torch.from_numpy(normal_vis.astype(np.float32)))
            foreground_rgb_results.append(torch.from_numpy(foreground_rgb))
            foreground_mask_results.append(torch.from_numpy(foreground_mask))
        
        # Stack results
        depth_output = torch.stack(depth_results).to(device)
        normal_output = torch.stack(normal_results).to(device)
        foreground_rgb_output = torch.stack(foreground_rgb_results).to(device)
        foreground_mask_output = torch.stack(foreground_mask_results).to(device)
        
        return (depth_output, normal_output, foreground_rgb_output, foreground_mask_output) 