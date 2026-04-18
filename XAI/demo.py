"""
Demo Script: 3D Attribution with Dummy Data
============================================

This script demonstrates the attribution toolkit using synthetic data.
Use this to verify your installation and understand the workflow.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from monai.networks.nets import UNet

# Import our attribution tools
from attribution_3d_heatmaps import (
    GradCAM3D, GradCAMPlusPlus3D, OcclusionSensitivity3D,
    Attribution3DVisualizer
)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-cc67418b-8d94-51c4-88d2-ec45ff0e92c0" # Adjust as needed

def create_dummy_model(device='cuda'):
    """Create a simple UNet model with random weights for demonstration."""
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    )
    model = model.to(device)
    model.eval()
    return model


def create_synthetic_volume():
    """Create a synthetic 3D volume with a sphere for testing."""
    # Create a 64x64x64 volume
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    
    # Add a sphere in the center
    center = (32, 32, 32)
    radius = 15
    
    for z in range(64):
        for y in range(64):
            for x in range(64):
                dist = np.sqrt(
                    (x - center[0])**2 + 
                    (y - center[1])**2 + 
                    (z - center[2])**2
                )
                if dist < radius:
                    volume[z, y, x] = 1.0
    
    # Add some noise
    volume += np.random.randn(64, 64, 64) * 0.1
    
    # Clip to [0, 1]
    volume = np.clip(volume, 0, 1)
    
    return volume


def main():
    """Run the complete demonstration."""
    print("=" * 60)
    print("3D Attribution Demo with Synthetic Data")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Create output directory
    output_dir = Path('demo_results')
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    # Step 1: Create model
    print("\n[1/6] Creating dummy model...")
    model = create_dummy_model(device)
    print("✓ Model created")
    
    # Step 2: Create synthetic data
    print("\n[2/6] Generating synthetic volume...")
    volume = create_synthetic_volume()
    input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)
    print(f"✓ Volume shape: {input_tensor.shape}")
    
    # Visualize input
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    slices = np.linspace(10, 54, 8, dtype=int)
    for i, slice_idx in enumerate(slices):
        axes[i].imshow(volume[slice_idx], cmap='gray')
        axes[i].set_title(f'Input Slice {slice_idx}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / 'input_volume.png', dpi=150)
    print(f"✓ Input visualization saved")
    
    # Step 3: Generate GradCAM
    print("\n[3/6] Generating GradCAM...")
    gradcam = GradCAM3D(model, target_layer='model.2.0.conv')
    gradcam_map = gradcam.generate_cam(input_tensor, target_class=1)
    print(f"✓ GradCAM shape: {gradcam_map.shape}")
    
    # Step 4: Generate GradCAM++
    print("\n[4/6] Generating GradCAM++...")
    gradcam_pp = GradCAMPlusPlus3D(model, target_layer='model.2.0.conv')
    gradcam_pp_map = gradcam_pp.generate_cam(input_tensor, target_class=1)
    print(f"✓ GradCAM++ shape: {gradcam_pp_map.shape}")
    
    # Step 5: Generate Occlusion Sensitivity (smaller patches for speed)
    print("\n[5/6] Generating Occlusion Sensitivity...")
    print("   (This may take a minute...)")
    occlusion = OcclusionSensitivity3D(
        model,
        patch_size=(8, 8, 8),
        stride=(4, 4, 4)
    )
    occlusion_map = occlusion.generate_map(input_tensor, target_class=1)
    print(f"✓ Occlusion map shape: {occlusion_map.shape}")
    
    # Step 6: Visualize results
    print("\n[6/6] Creating visualizations...")
    visualizer = Attribution3DVisualizer()
    
    # GradCAM slices
    visualizer.plot_slices(
        volume,
        gradcam_map,
        num_slices=9,
        axis=0,
        alpha=0.5,
        save_path=output_dir / 'gradcam_slices.png'
    )
    print("✓ GradCAM visualization saved")
    
    # GradCAM++ slices
    visualizer.plot_slices(
        volume,
        gradcam_pp_map,
        num_slices=9,
        axis=0,
        alpha=0.5,
        save_path=output_dir / 'gradcam_pp_slices.png'
    )
    print("✓ GradCAM++ visualization saved")
    
    # Occlusion slices
    visualizer.plot_slices(
        volume,
        occlusion_map,
        num_slices=9,
        axis=0,
        alpha=0.5,
        save_path=output_dir / 'occlusion_slices.png'
    )
    print("✓ Occlusion visualization saved")
    
    # Comparison view
    middle_slice = 32
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Top row: overlays
    axes[0, 0].imshow(volume[middle_slice], cmap='gray')
    axes[0, 0].imshow(gradcam_map[middle_slice], alpha=0.5, cmap='jet')
    axes[0, 0].set_title('GradCAM Overlay', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(volume[middle_slice], cmap='gray')
    axes[0, 1].imshow(gradcam_pp_map[middle_slice], alpha=0.5, cmap='jet')
    axes[0, 1].set_title('GradCAM++ Overlay', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(volume[middle_slice], cmap='gray')
    axes[0, 2].imshow(occlusion_map[middle_slice], alpha=0.5, cmap='jet')
    axes[0, 2].set_title('Occlusion Overlay', fontsize=14)
    axes[0, 2].axis('off')
    
    # Bottom row: heatmaps only
    im1 = axes[1, 0].imshow(gradcam_map[middle_slice], cmap='jet')
    axes[1, 0].set_title('GradCAM Heatmap', fontsize=14)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    im2 = axes[1, 1].imshow(gradcam_pp_map[middle_slice], cmap='jet')
    axes[1, 1].set_title('GradCAM++ Heatmap', fontsize=14)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    im3 = axes[1, 2].imshow(occlusion_map[middle_slice], cmap='jet')
    axes[1, 2].set_title('Occlusion Heatmap', fontsize=14)
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
    
    plt.suptitle('Attribution Comparison - Middle Slice', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison visualization saved")
    
    # Volume projections
    visualizer.create_volume_projection(
        gradcam_map,
        method='max',
        save_path=output_dir / 'projections.png'
    )
    print("✓ Projection visualization saved")
    
    # Save attribution maps
    np.save(output_dir / 'gradcam.npy', gradcam_map)
    np.save(output_dir / 'gradcam_pp.npy', gradcam_pp_map)
    np.save(output_dir / 'occlusion.npy', occlusion_map)
    print("✓ Attribution maps saved as .npy files")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo Complete! ✓")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  • {file.name}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review the generated visualizations")
    print("2. Modify the script to use your own model and data")
    print("3. Adjust target_layer based on your model architecture")
    print("4. Experiment with different attribution methods")
    print("5. Check out attribution_analysis.ipynb for interactive analysis")
    print("\nFor more information, see README.md")
    print("=" * 60)


if __name__ == '__main__':
    main()
