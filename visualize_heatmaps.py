#!/usr/bin/env python3
"""
Simple script to load a dataset, generate heatmaps, and visualize them with matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from utils import generate_heatmap

def load_and_visualize_heatmaps(dataset_path, episode_id=0, timestep=0):
    """
    Load a dataset episode and visualize the heatmaps
    
    Args:
        dataset_path: Path to the dataset directory
        episode_id: Episode ID to load
        timestep: Timestep within the episode
    """
    
    # Construct file path
    episode_file = os.path.join(dataset_path, f'episode_{episode_id}.hdf5')
    
    if not os.path.exists(episode_file):
        print(f"Episode file not found: {episode_file}")
        return
    
    print(f"Loading episode {episode_id} from {episode_file}")
    
    with h5py.File(episode_file, 'r') as root:
        # Load images
        camera_names = ['ap', 'lateral']
        images = {}
        for cam_name in camera_names:
            if f'/observations/images/{cam_name}' in root:
                images[cam_name] = root[f'/observations/images/{cam_name}'][timestep]
                print(f"Loaded {cam_name} image with shape: {images[cam_name].shape}")
            else:
                print(f"Warning: Camera {cam_name} not found in dataset")
        
        # Load annotations
        if '/annotations/start' in root and '/annotations/end' in root:
            annotation_start = root['/annotations/start'][()]
            annotation_end = root['/annotations/end'][()]
            print(f"Annotation start: {annotation_start}")
            print(f"Annotation end: {annotation_end}")
        else:
            print("Warning: Annotations not found in dataset")
            return
        
        if '/world_from_anatomical' in root:
            world = root['/world_from_anatomical'][()]
            print(f"Loaded world from anatomical matrix with shape: {world.shape}")
        else:
            print("Warning: World from anatomical matrix not found in dataset")
            return
        
        # Load projection matrices
        projection_matrices = {}
        for cam_name in camera_names:
            proj_path = f'/observations/projection_matrices/{cam_name}'
            if proj_path in root:
                projection_matrices[cam_name] = root[proj_path][()]
                print(f"Loaded projection matrix for {cam_name}: {projection_matrices[cam_name].shape}")
            else:
                print(f"Warning: Projection matrix for {cam_name} not found")
    
    # Generate heatmaps
    heatmaps = {}
    for cam_name in camera_names:
        if cam_name in images and cam_name in projection_matrices:
            print(f"Generating heatmap for {cam_name}...")
            heatmap = generate_heatmap(
                np.asarray(annotation_start, dtype=np.float32), 
                np.asarray(annotation_end, dtype=np.float32), 
                projection_matrices[cam_name], 
                images[cam_name].shape[:2],  # (H, W)
                world_from_anatomical=world
            )
            heatmaps[cam_name] = np.asarray(heatmap)
            print(f"Generated heatmap for {cam_name} with shape: {heatmap.shape}")
            print(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Visualize results
    visualize_images_and_heatmaps(images, heatmaps, episode_id, timestep)

def visualize_images_and_heatmaps(images, heatmaps, episode_id, timestep):
    """
    Create visualization plots showing original images, heatmaps, and overlays
    """
    n_cameras = len(images)
    if n_cameras == 0:
        print("No images to visualize")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, n_cameras, figsize=(5*n_cameras, 12))
    if n_cameras == 1:
        axes = axes.reshape(-1, 1)
    
    camera_names = list(images.keys())
    
    for i, cam_name in enumerate(camera_names):
        img = images[cam_name]
        heatmap = heatmaps.get(cam_name, None)
        
        # Handle different image formats (RGB vs grayscale)
        if len(img.shape) == 3 and img.shape[2] == 3:
            # RGB image
            img_display = img
        else:
            # Grayscale image
            img_display = img
        
        # 1. Original image
        img_plot = axes[0, i].imshow(img_display, cmap='gray' if len(img.shape) == 2 else None)
        axes[0, i].set_title(f'{cam_name.upper()} - Original Image')
        axes[0, i].axis('off')
        cbar = plt.colorbar(img_plot, ax=axes[0, i], fraction=0.046, pad=0.04, cmap='gray')
        cbar.ax.set_visible(False)
        
        if heatmap is not None:
            # 2. Heatmap
            heatmap_plot = axes[1, i].imshow(heatmap, cmap='hot', alpha=0.8)
            axes[1, i].set_title(f'{cam_name.upper()} - Heatmap')
            axes[1, i].axis('off')
            plt.colorbar(heatmap_plot, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            # 3. Overlay
            axes[2, i].imshow(img_display, cmap='gray' if len(img.shape) == 2 else None)
            overlay = axes[2, i].imshow(heatmap, cmap='hot', alpha=0.5)
            axes[2, i].set_title(f'{cam_name.upper()} - Overlay')
            axes[2, i].axis('off')
            plt.colorbar(overlay, ax=axes[2, i], fraction=0.046, pad=0.04)
        else:
            # No heatmap available
            axes[1, i].text(0.5, 0.5, 'No heatmap\navailable', 
                           ha='center', va='center', transform=axes[1, i].transAxes,
                           fontsize=14)
            axes[1, i].axis('off')
            axes[2, i].text(0.5, 0.5, 'No heatmap\navailable', 
                           ha='center', va='center', transform=axes[2, i].transAxes,
                           fontsize=14)
            axes[2, i].axis('off')
    
    plt.suptitle(f'Episode {episode_id}, Timestep {timestep} - Heatmap Visualization', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    output_path = f'heatmap_visualization_ep{episode_id}_t{timestep}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    
    # Show the plot
    # plt.savefig("test.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

load_and_visualize_heatmaps("/data2/flora/vertebroplasty_imitation_0", 190, 0)
print("\nVisualization completed successfully!")
