#!/usr/bin/env python3
"""
Small script to visualize dataloader samples by saving individual channel images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data
from constants import SIM_TASK_CONFIGS

def save_channel_images(image_batch, sample_idx, dataloader_type, output_dir, camera_names):
    """
    Save each channel of an image batch as separate plots.
    
    Args:
        image_batch: Tensor of shape (batch_size, cameras, channels, height, width)
        sample_idx: Index of the sample in the batch
        dataloader_type: 'val' or 'test' for naming
        output_dir: Directory to save images
        camera_names: List of camera names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get single sample: (cameras, channels, height, width)
    sample = image_batch[sample_idx]
    
    for cam_idx, cam_name in enumerate(camera_names):
        for channel_idx in range(sample.shape[1]):  # Iterate through channels
            # Extract single channel image
            channel_img = sample[cam_idx, channel_idx].cpu().numpy()
            
            # Create plot
            plt.figure(figsize=(6, 6))
            plt.imshow(channel_img, cmap='gray')
            plt.title(f'{dataloader_type.upper()} - {cam_name} - Channel {channel_idx}')
            plt.axis('off')
            
            # Save image
            filename = f'{dataloader_type}_sample{sample_idx}_{cam_name}_ch{channel_idx}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=150)
            plt.close()
            
            print(f"Saved: {filepath}")

def main():
    # Use NMDID_v2.4 configuration from constants
    task_name = 'NMDID_v2.4'
    task_config = SIM_TASK_CONFIGS[task_name]
    
    # Configuration from constants
    dataset_dir = task_config['dataset_dir']
    train_dir = task_config['train_dir']
    val_dir = task_config['val_dir']
    test_dir = task_config['test_dir']
    camera_names = task_config['camera_names']
    num_episodes = task_config['num_episodes']
    episodes_start = task_config['episode_start']
    
    batch_size = 4
    output_dir = "/home/flora/projects/verteboplasty_imitation/external/act/xray_transforms_image_tests"
    
    print(f"Using task configuration: {task_name}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Camera names: {camera_names}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print(f"Test dir: {test_dir}")
    
    print("Loading validation dataloader...")
    # Create validation dataloader
    _, val_dataloader, _, _ = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        episodes_start=episodes_start,
        camera_names=camera_names,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        train_dir=train_dir,
        val_dir=val_dir
    )
    
    print("Loading test dataloader...")
    # Create test dataloader using actual test directory
    _, test_dataloader, _, _ = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        episodes_start=episodes_start,
        camera_names=camera_names,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        train_dir=train_dir,  # Use train_dir for normalization stats
        val_dir=test_dir     # Use test_dir as validation for test dataloader
    )
    
    print("\nProcessing validation samples...")
    # Get a few samples from validation dataloader
    val_batch = next(iter(val_dataloader))
    image_data, qpos_data, action_data, is_pad = val_batch
    
    print(f"Val batch shapes - Images: {image_data.shape}, QPos: {qpos_data.shape}")
    print(f"Number of cameras: {len(camera_names)}, Channels per camera: {image_data.shape[2]}")
    
    # Save first 2 samples from validation
    for i in range(min(2, image_data.shape[0])):
        save_channel_images(image_data, i, 'val', output_dir, camera_names)
    
    print("\nProcessing test samples...")
    # Get a few samples from test dataloader
    test_batch = next(iter(test_dataloader))
    image_data, qpos_data, action_data, is_pad = test_batch
    
    print(f"Test batch shapes - Images: {image_data.shape}, QPos: {qpos_data.shape}")
    print(f"Number of cameras: {len(camera_names)}, Channels per camera: {image_data.shape[2]}")
    
    # Save first 2 samples from test
    for i in range(min(2, image_data.shape[0])):
        save_channel_images(image_data, i, 'test', output_dir, camera_names)
    
    print(f"\nAll images saved to: {output_dir}")
    print("Image naming convention: [val/test]_sample[idx]_[camera]_ch[channel].png")

if __name__ == "__main__":
    main()
