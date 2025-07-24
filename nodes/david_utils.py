"""
Utility nodes for DAViD outputs
"""

import torch
import numpy as np
import cv2

class DAViDDepthVisualizer:
    """
    Convert raw depth values to various visualization formats
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_raw": ("IMAGE",),  # Expected to be single channel depth
                "colormap": (["TURBO", "JET", "HSV", "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "BONE", "GRAY"], {"default": "TURBO"}),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
                "normalize": ("BOOLEAN", {"default": True}),
                "min_depth": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "max_depth": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_visualization",)
    FUNCTION = "visualize"
    CATEGORY = "DAViD/Utils"
    
    def visualize(self, depth_raw, colormap="TURBO", invert=False, normalize=True, min_depth=0.0, max_depth=1.0):
        """Convert raw depth to colored visualization"""
        
        # Colormap mapping
        colormap_dict = {
            "TURBO": cv2.COLORMAP_TURBO,
            "JET": cv2.COLORMAP_JET,
            "HSV": cv2.COLORMAP_HSV,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "BONE": cv2.COLORMAP_BONE,
            "GRAY": None  # Special case for grayscale
        }
        
        batch_size = depth_raw.shape[0]
        results = []
        
        for i in range(batch_size):
            depth = depth_raw[i].cpu().numpy()
            
            # If depth is RGB, convert to single channel
            if len(depth.shape) == 3 and depth.shape[2] == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
            elif len(depth.shape) == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            
            # Invert if requested
            if invert:
                depth = -depth
            
            # Normalize
            if normalize:
                if abs(max_depth - min_depth) > 0.001:
                    depth = (depth - min_depth) / (max_depth - min_depth)
                else:
                    depth = np.clip(depth, 0, 1)
            else:
                depth = np.clip(depth, 0, 1)
            
            # Convert to uint8
            depth_uint8 = (depth * 255).astype(np.uint8)
            
            # Apply colormap
            if colormap == "GRAY":
                # For grayscale, just stack to RGB
                depth_colored = np.stack([depth] * 3, axis=-1)
            else:
                # Apply OpenCV colormap
                depth_colored_bgr = cv2.applyColorMap(depth_uint8, colormap_dict[colormap])
                depth_colored = cv2.cvtColor(depth_colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            results.append(torch.from_numpy(depth_colored))
        
        return (torch.stack(results),)


class DAViDNormalToLight:
    """
    Convert surface normals to lighting/shading
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "light_direction": ("STRING", {"default": "0.5,0.5,1.0", "multiline": False}),
                "ambient_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "diffuse_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("shaded_image",)
    FUNCTION = "apply_lighting"
    CATEGORY = "DAViD/Utils"
    
    def apply_lighting(self, normal_map, light_direction="0.5,0.5,1.0", ambient_strength=0.3, diffuse_strength=0.7):
        """Apply lighting to surface normals"""
        
        # Parse light direction
        try:
            light_dir = np.array([float(x) for x in light_direction.split(',')])
            light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
        except:
            light_dir = np.array([0.5, 0.5, 1.0])
            light_dir = light_dir / np.linalg.norm(light_dir)
        
        batch_size = normal_map.shape[0]
        results = []
        
        for i in range(batch_size):
            normals = normal_map[i].cpu().numpy()
            
            # Convert from visualization space [0,1] back to normal space [-1,1]
            normals = normals * 2.0 - 1.0
            
            # Normalize
            normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8)
            
            # Calculate diffuse lighting (Lambertian)
            diffuse = np.maximum(0, np.dot(normals, light_dir))
            
            # Combine ambient and diffuse
            shading = ambient_strength + diffuse_strength * diffuse
            shading = np.clip(shading, 0, 1)
            
            # Convert to RGB
            shaded = np.stack([shading] * 3, axis=-1)
            
            results.append(torch.from_numpy(shaded.astype(np.float32)))
        
        return (torch.stack(results),) 