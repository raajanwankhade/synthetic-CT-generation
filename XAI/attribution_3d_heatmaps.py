"""
3D Attribution and Voxel-Level Heatmaps for MONAI UNET3D
===========================================================

This script provides comprehensive tools for generating attribution heatmaps
for 3D medical imaging models, specifically UNET3D from MONAI.

Features:
- GradCAM, GradCAM++, and Occlusion Sensitivity
- 3D voxel-level heatmaps
- Interactive visualizations
- Support for multi-class segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from pathlib import Path

from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRanged, EnsureTyped
)
from monai.data import DataLoader, Dataset
from monai.visualize import blend_images


class GradCAM3D:
    """
    3D GradCAM implementation for volumetric medical imaging models.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize GradCAM.
        
        Args:
            model: Trained MONAI UNet model
            target_layer: Name of the layer to compute gradients for
                         (e.g., 'model.0.submodule.5' for last encoder block)
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Navigate to target layer
        target = self.model
        for attr in self.target_layer.split('.'):
            target = getattr(target, attr)
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_slice: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate 3D GradCAM heatmap.
        
        Args:
            input_tensor: Input volume [B, C, D, H, W]
            target_class: Class index for segmentation models
            target_slice: Optional specific slice to focus on
        
        Returns:
            3D heatmap of same spatial size as input
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For segmentation: if target_class specified, use it
        # Otherwise, use the predicted class
        if target_class is not None:
            if len(output.shape) == 5:  # [B, C, D, H, W]
                target = output[:, target_class, :, :, :]
            else:
                target = output
        else:
            # Use max predicted value
            target = output.max()
        
        # Backward pass (sum to ensure scalar for backwards)
        target.sum().backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, D, H, W]
        activations = self.activations  # [B, C, D, H, W]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, D, H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size if needed
        if cam.shape != input_tensor.shape[2:]:
            cam = self._resize_3d(cam, input_tensor.shape[2:])
        
        return cam
    
    def _resize_3d(self, cam: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D array to target shape using trilinear interpolation."""
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            cam_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        return resized.squeeze().numpy()


class GradCAMPlusPlus3D(GradCAM3D):
    """
    3D GradCAM++ implementation with improved localization.
    """
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_slice: Optional[int] = None
    ) -> np.ndarray:
        """Generate 3D GradCAM++ heatmap."""
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is not None:
            if len(output.shape) == 5:
                target = output[:, target_class, :, :, :]
            else:
                target = output
        else:
            target = output.max()
        
        target.sum().backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # GradCAM++ weights
        grad_squared = gradients ** 2
        grad_cubed = grad_squared * gradients
        
        alpha_num = grad_squared
        alpha_denom = 2 * grad_squared + (grad_cubed * activations).sum(dim=(2, 3, 4), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        
        alpha = alpha_num / alpha_denom
        
        # Weights with ReLU applied to gradients
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3, 4), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        if cam.shape != input_tensor.shape[2:]:
            cam = self._resize_3d(cam, input_tensor.shape[2:])
        
        return cam


class OcclusionSensitivity3D:
    """
    3D Occlusion Sensitivity for understanding model decisions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        stride: Tuple[int, int, int] = (8, 8, 8)
    ):
        """
        Initialize Occlusion Sensitivity.
        
        Args:
            model: Trained model
            patch_size: Size of occlusion patch (depth, height, width)
            stride: Stride for sliding the occlusion patch
        """
        self.model = model
        self.model.eval()
        self.patch_size = patch_size
        self.stride = stride
    
    def generate_map(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        occlusion_value: float = 0.0
    ) -> np.ndarray:
        """
        Generate 3D occlusion sensitivity map.
        
        Args:
            input_tensor: Input volume [B, C, D, H, W]
            target_class: Target class for evaluation
            occlusion_value: Value to use for occluded regions
        
        Returns:
            3D sensitivity map
        """
        B, C, D, H, W = input_tensor.shape
        device = input_tensor.device
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            if target_class is not None and len(baseline_output.shape) == 5:
                baseline_score = baseline_output[:, target_class, :, :, :].mean().item()
            else:
                baseline_score = baseline_output.mean().item()
        
        # Initialize sensitivity map
        sensitivity_map = np.zeros((D, H, W))
        count_map = np.zeros((D, H, W))
        
        # Slide occlusion patch
        for d in range(0, D - self.patch_size[0] + 1, self.stride[0]):
            for h in range(0, H - self.patch_size[1] + 1, self.stride[1]):
                for w in range(0, W - self.patch_size[2] + 1, self.stride[2]):
                    # Create occluded input
                    occluded_input = input_tensor.clone()
                    occluded_input[
                        :, :,
                        d:d+self.patch_size[0],
                        h:h+self.patch_size[1],
                        w:w+self.patch_size[2]
                    ] = occlusion_value
                    
                    # Get prediction
                    with torch.no_grad():
                        output = self.model(occluded_input)
                        if target_class is not None and len(output.shape) == 5:
                            score = output[:, target_class, :, :, :].mean().item()
                        else:
                            score = output.mean().item()
                    
                    # Calculate sensitivity (drop in score)
                    sensitivity = baseline_score - score
                    
                    # Update maps
                    sensitivity_map[
                        d:d+self.patch_size[0],
                        h:h+self.patch_size[1],
                        w:w+self.patch_size[2]
                    ] += sensitivity
                    
                    count_map[
                        d:d+self.patch_size[0],
                        h:h+self.patch_size[1],
                        w:w+self.patch_size[2]
                    ] += 1
        
        # Average overlapping regions
        sensitivity_map = sensitivity_map / (count_map + 1e-8)
        
        # Normalize
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / \
                         (sensitivity_map.max() - sensitivity_map.min() + 1e-8)
        
        return sensitivity_map


class Attribution3DVisualizer:
    """
    Visualization tools for 3D attribution heatmaps.
    """
    
    @staticmethod
    def plot_slices(
        volume: np.ndarray,
        heatmap: np.ndarray,
        num_slices: int = 9,
        axis: int = 0,
        alpha: float = 0.5,
        cmap: str = 'jet',
        save_path: Optional[str] = None
    ):
        """
        Plot multiple slices with overlaid heatmaps.
        
        Args:
            volume: Original 3D volume [D, H, W]
            heatmap: 3D heatmap [D, H, W]
            num_slices: Number of slices to display
            axis: Axis along which to slice (0=depth, 1=height, 2=width)
            alpha: Transparency of heatmap overlay
            cmap: Colormap for heatmap
            save_path: Optional path to save the figure
        """
        # Select evenly spaced slices
        indices = np.linspace(0, volume.shape[axis] - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices):
            if axis == 0:
                img_slice = volume[idx, :, :]
                heat_slice = heatmap[idx, :, :]
            elif axis == 1:
                img_slice = volume[:, idx, :]
                heat_slice = heatmap[:, idx, :]
            else:
                img_slice = volume[:, :, idx]
                heat_slice = heatmap[:, :, idx]
            
            # Plot original image
            axes[i].imshow(img_slice, cmap='gray')
            
            # Overlay heatmap
            axes[i].imshow(heat_slice, cmap=cmap, alpha=alpha)
            
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_3d_isosurface(
        heatmap: np.ndarray,
        threshold: float = 0.5,
        save_path: Optional[str] = None
    ):
        """
        Create 3D isosurface visualization of heatmap.
        Requires plotly for interactive visualization.
        """
        try:
            import plotly.graph_objects as go
            from skimage import measure
        except ImportError:
            print("Please install plotly and scikit-image for 3D visualization:")
            print("pip install plotly scikit-image")
            return
        
        # Create isosurface
        verts, faces, _, _ = measure.marching_cubes(heatmap, level=threshold)
        
        # Create mesh
        x, y, z = verts.T
        i, j, k = faces.T
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                intensity=heatmap[verts[:, 0].astype(int),
                                 verts[:, 1].astype(int),
                                 verts[:, 2].astype(int)],
                colorscale='Jet',
                opacity=0.8,
                name='Attribution'
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='3D Attribution Heatmap'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    @staticmethod
    def create_volume_projection(
        heatmap: np.ndarray,
        method: str = 'max',
        save_path: Optional[str] = None
    ):
        """
        Create maximum/average intensity projections along each axis.
        
        Args:
            heatmap: 3D heatmap
            method: 'max' or 'mean' projection
            save_path: Optional save path
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if method == 'max':
            proj_func = np.max
        else:
            proj_func = np.mean
        
        # Projections along each axis
        projections = [
            proj_func(heatmap, axis=0),  # Along depth
            proj_func(heatmap, axis=1),  # Along height
            proj_func(heatmap, axis=2)   # Along width
        ]
        
        titles = ['Depth Projection', 'Height Projection', 'Width Projection']
        
        for ax, proj, title in zip(axes, projections, titles):
            im = ax.imshow(proj, cmap='jet')
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def load_model(
    model_path: str,
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: Tuple[int, ...] = (2, 2, 2, 2),
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load a trained UNet model.
    
    Args:
        model_path: Path to model weights (.pth file)
        spatial_dims: Number of spatial dimensions (3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels/classes
        channels: Channel sequence for encoder
        strides: Stride sequence
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=2
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle different state dict formats
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    elif 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    return model


def main_example():
    """
    Example usage of the attribution tools.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================
    # 1. LOAD YOUR MODEL
    # ========================================
    print("\n1. Loading model...")
    model = load_model(
        model_path='path/to/your/model_weights.pth',  # UPDATE THIS
        spatial_dims=3,
        in_channels=1,
        out_channels=2,  # UPDATE based on your task
        device=device
    )
    print("Model loaded successfully!")
    
    # ========================================
    # 2. PREPARE YOUR INPUT
    # ========================================
    print("\n2. Preparing input...")
    # Option A: Load from file
    # transforms = Compose([
    #     LoadImaged(keys=["image"]),
    #     EnsureChannelFirstd(keys=["image"]),
    #     Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    #     ScaleIntensityRanged(
    #         keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
    #     ),
    #     EnsureTyped(keys=["image"]),
    # ])
    # data = transforms({"image": "path/to/your/image.nii.gz"})
    # input_tensor = data["image"].unsqueeze(0).to(device)
    
    # Option B: Use dummy data for testing
    input_tensor = torch.randn(1, 1, 64, 128, 128).to(device)
    print(f"Input shape: {input_tensor.shape}")
    
    # ========================================
    # 3. GENERATE ATTRIBUTION MAPS
    # ========================================
    
    # GradCAM
    print("\n3. Generating GradCAM...")
    # For UNet, typically use the last encoder block
    # Adjust layer name based on your model architecture
    gradcam = GradCAM3D(model, target_layer='model.0.submodule.4')
    cam_map = gradcam.generate_cam(input_tensor, target_class=1)
    print(f"GradCAM shape: {cam_map.shape}")
    
    # GradCAM++
    print("\n4. Generating GradCAM++...")
    gradcam_pp = GradCAMPlusPlus3D(model, target_layer='model.0.submodule.4')
    cam_pp_map = gradcam_pp.generate_cam(input_tensor, target_class=1)
    print(f"GradCAM++ shape: {cam_pp_map.shape}")
    
    # Occlusion Sensitivity (slower but more interpretable)
    print("\n5. Generating Occlusion Sensitivity Map...")
    print("   (This may take a while...)")
    occlusion = OcclusionSensitivity3D(
        model,
        patch_size=(8, 16, 16),
        stride=(4, 8, 8)
    )
    occ_map = occlusion.generate_map(input_tensor, target_class=1)
    print(f"Occlusion map shape: {occ_map.shape}")
    
    # ========================================
    # 4. VISUALIZE RESULTS
    # ========================================
    print("\n6. Creating visualizations...")
    
    visualizer = Attribution3DVisualizer()
    
    # Prepare volume for visualization
    volume = input_tensor[0, 0].cpu().numpy()
    
    # Plot slices with GradCAM
    visualizer.plot_slices(
        volume,
        cam_map,
        num_slices=9,
        axis=0,
        save_path='gradcam_slices.png'
    )
    
    # Plot slices with GradCAM++
    visualizer.plot_slices(
        volume,
        cam_pp_map,
        num_slices=9,
        axis=0,
        save_path='gradcam_pp_slices.png'
    )
    
    # Plot slices with Occlusion
    visualizer.plot_slices(
        volume,
        occ_map,
        num_slices=9,
        axis=0,
        save_path='occlusion_slices.png'
    )
    
    # Create projections
    visualizer.create_volume_projection(
        cam_map,
        method='max',
        save_path='gradcam_projections.png'
    )
    
    # 3D isosurface (optional, requires plotly)
    try:
        visualizer.plot_3d_isosurface(
            cam_map,
            threshold=0.6,
            save_path='gradcam_3d.html'
        )
    except:
        print("Skipping 3D visualization (requires plotly)")
    
    print("\n✓ Attribution analysis complete!")
    print("Check the generated PNG and HTML files for visualizations.")
    
    return {
        'gradcam': cam_map,
        'gradcam++': cam_pp_map,
        'occlusion': occ_map,
        'volume': volume
    }


if __name__ == '__main__':
    results = main_example()
