"""
ComfyUI-DAViD: Custom nodes for DAViD models
Provides depth estimation, surface normal estimation, and foreground segmentation
"""

from .nodes import DAViDMultiTaskNode, DAViDDepthVisualizer, DAViDNormalToLight

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "DAViDMultiTask": DAViDMultiTaskNode,
    "DAViDDepthVisualizer": DAViDDepthVisualizer,
    "DAViDNormalToLight": DAViDNormalToLight,
}

# Display names for nodes in ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "DAViDMultiTask": "DAViD Multi-Task (Depth/Normal/Foreground)",
    "DAViDDepthVisualizer": "DAViD Depth Visualizer",
    "DAViDNormalToLight": "DAViD Normal to Lighting",
}

# Web directory (if you have custom web components)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"] 