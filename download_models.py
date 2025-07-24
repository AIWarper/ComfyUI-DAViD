#!/usr/bin/env python3
"""
Download script for DAViD models
"""

import os
import sys
import urllib.request
from pathlib import Path

# Model URLs
MODELS = {
    "multitask-vitl16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/multi-task-model-vitl16_384.onnx",
    "depth-vitb16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/depth-model-vitb16_384.onnx",
    "depth-vitl16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/depth-model-vitl16_384.onnx",
    "normal-vitb16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/normal-model-vitb16_384.onnx",
    "normal-vitl16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/normal-model-vitl16_384.onnx",
    "foreground-vitb16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitb16_384.onnx",
    "foreground-vitl16_384.onnx": "https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/foreground-segmentation-model-vitl16_384.onnx",
}

def download_file(url, filepath):
    """Download a file with progress reporting"""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    
    print(f"Downloading {os.path.basename(filepath)}...")
    urllib.request.urlretrieve(url, filepath, download_progress)
    print()  # New line after progress

def main():
    # Try to find ComfyUI models directory
    if os.path.exists("../../models"):
        # We're in custom_nodes/ComfyUI-DAViD
        models_dir = Path("../../models/david")
    else:
        # Use local models directory
        models_dir = Path("models/david")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("DAViD Model Downloader")
    print("=" * 50)
    print(f"Models will be saved to: {models_dir.absolute()}")
    print()
    
    # Check which models are already downloaded
    existing_models = []
    missing_models = []
    
    for model_name in MODELS.keys():
        model_path = models_dir / model_name
        if model_path.exists():
            existing_models.append(model_name)
        else:
            missing_models.append(model_name)
    
    if existing_models:
        print("Already downloaded:")
        for model in existing_models:
            print(f"  ✓ {model}")
        print()
    
    if not missing_models:
        print("All models are already downloaded!")
        return
    
    print("Available models to download:")
    print("  1. multitask-vitl16_384.onnx (Recommended - all tasks in one model)")
    print("  2. All models")
    for i, model in enumerate(missing_models, 3):
        if model.startswith("multitask"):
            continue
        print(f"  {i}. {model}")
    
    print()
    choice = input("Enter your choice (1-9) or 'q' to quit: ").strip()
    
    if choice.lower() == 'q':
        return
    
    models_to_download = []
    
    if choice == '1':
        if "multitask-vitl16_384.onnx" in missing_models:
            models_to_download = ["multitask-vitl16_384.onnx"]
    elif choice == '2':
        models_to_download = missing_models
    else:
        try:
            idx = int(choice) - 3
            if 0 <= idx < len(missing_models) - 1:
                models_to_download = [missing_models[idx + 1]]  # +1 to skip multitask
        except:
            print("Invalid choice!")
            return
    
    if not models_to_download:
        print("Model already downloaded or invalid choice!")
        return
    
    print()
    for model_name in models_to_download:
        model_path = models_dir / model_name
        try:
            download_file(MODELS[model_name], str(model_path))
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    print()
    print("Download complete!")

if __name__ == "__main__":
    main() 