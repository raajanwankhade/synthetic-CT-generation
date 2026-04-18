# 3D Attribution and Voxel-Level Heatmaps for MONAI

A comprehensive toolkit for generating and visualizing attribution heatmaps for 3D medical imaging models (UNET3D) using MONAI.

## Features

- **Multiple Attribution Methods**:
  - GradCAM: Gradient-weighted Class Activation Mapping
  - GradCAM++: Improved localization with better multi-instance support
  - Occlusion Sensitivity: Model-agnostic approach using sliding occlusion windows

- **Rich Visualizations**:
  - Multi-slice 2D visualizations with heatmap overlays
  - 3D isosurface rendering (interactive with Plotly)
  - Maximum/Average intensity projections
  - Comparison views across methods

- **Export Formats**:
  - NumPy arrays (.npy)
  - NIfTI medical imaging format (.nii.gz)
  - PNG images for publications
  - Interactive HTML for 3D exploration

## Installation

```bash
# Core requirements
pip install torch torchvision
pip install monai
pip install numpy matplotlib

# Optional for enhanced visualizations
pip install plotly scikit-image nibabel

# Or install all at once
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage (Python Script)

```python
from attribution_3d_heatmaps import (
    GradCAM3D, Attribution3DVisualizer, load_model
)
import torch

# Load your trained model
model = load_model(
    model_path='path/to/your/model_weights.pth',
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    device='cuda'
)

# Prepare your input (shape: [1, 1, D, H, W])
input_tensor = torch.randn(1, 1, 64, 128, 128).cuda()

# Generate GradCAM heatmap
gradcam = GradCAM3D(model, target_layer='model.0.submodule.4')
heatmap = gradcam.generate_cam(input_tensor, target_class=1)

# Visualize
visualizer = Attribution3DVisualizer()
volume = input_tensor[0, 0].cpu().numpy()
visualizer.plot_slices(volume, heatmap, num_slices=9)
```

### 2. Using the Jupyter Notebook

Open `attribution_analysis.ipynb` for an interactive environment with:
- Step-by-step guidance
- Pre-configured visualizations
- Batch processing examples
- Export utilities

```bash
jupyter notebook attribution_analysis.ipynb
```

## Key Parameters to Configure

### Model Loading

```python
model = load_model(
    model_path='your_model.pth',
    spatial_dims=3,              # Always 3 for 3D models
    in_channels=1,               # Typically 1 for CT/MRI
    out_channels=2,              # Number of segmentation classes
    channels=(16, 32, 64, 128, 256),  # Must match training
    strides=(2, 2, 2, 2),       # Must match training
    device='cuda'
)
```

### Target Layer Selection

For UNet models, common target layers:
- `'model.0.submodule.4'`: Last encoder block (most common)
- `'model.0.submodule.3'`: Second-to-last encoder block
- `'model.1.submodule.0'`: First decoder block

**How to find the right layer:**

```python
# Print model architecture
print(model)

# Or get all layer names
for name, module in model.named_modules():
    print(name)
```

Choose the deepest encoder layer or early decoder layer for best results.

### Attribution Method Selection

**GradCAM** - Fast, gradient-based:
```python
gradcam = GradCAM3D(model, target_layer='model.0.submodule.4')
heatmap = gradcam.generate_cam(input_tensor, target_class=1)
```

**GradCAM++** - Better for multiple instances:
```python
gradcam_pp = GradCAMPlusPlus3D(model, target_layer='model.0.submodule.4')
heatmap = gradcam_pp.generate_cam(input_tensor, target_class=1)
```

**Occlusion Sensitivity** - Slower but model-agnostic:
```python
occlusion = OcclusionSensitivity3D(
    model,
    patch_size=(8, 16, 16),  # Smaller = higher resolution but slower
    stride=(4, 8, 8)          # Smaller = more overlap, smoother results
)
heatmap = occlusion.generate_map(input_tensor, target_class=1)
```

## Understanding the Results

### Heatmap Interpretation

- **Red/Hot colors**: Regions most important for the prediction
- **Blue/Cold colors**: Regions less relevant
- **Intensity**: Relative importance (normalized 0-1)

### Which Method to Use?

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| GradCAM | Fast | Good | Quick analysis, single objects |
| GradCAM++ | Fast | Better | Multiple instances, complex scenes |
| Occlusion | Slow | Most reliable | Publication, verification |

**Recommendation**: Start with GradCAM for exploration, use Occlusion Sensitivity for final verification.

## Visualization Options

### 2D Slice Visualization

```python
visualizer.plot_slices(
    volume,
    heatmap,
    num_slices=9,      # How many slices to show
    axis=0,            # 0=depth, 1=height, 2=width
    alpha=0.5,         # Heatmap transparency
    cmap='jet',        # Colormap
    save_path='output.png'
)
```

### 3D Interactive Visualization

```python
visualizer.plot_3d_isosurface(
    heatmap,
    threshold=0.6,     # Show regions with attribution > 0.6
    save_path='3d_heatmap.html'
)
```

### Volume Projections

```python
visualizer.create_volume_projection(
    heatmap,
    method='max',      # 'max' or 'mean'
    save_path='projections.png'
)
```

## Common Issues and Solutions

### 1. CUDA Out of Memory

**For large volumes:**
```python
# Option 1: Use smaller patch size for occlusion
occlusion = OcclusionSensitivity3D(
    model,
    patch_size=(4, 8, 8),  # Reduced from (8, 16, 16)
    stride=(2, 4, 4)
)

# Option 2: Process smaller crops
crop_size = (64, 128, 128)  # Instead of full volume
input_crop = input_tensor[:, :, :crop_size[0], :crop_size[1], :crop_size[2]]
```

### 2. Target Layer Not Found

```python
# List all available layers
for name, _ in model.named_modules():
    if 'submodule' in name:
        print(name)

# Then use the correct name
gradcam = GradCAM3D(model, target_layer='YOUR_LAYER_NAME')
```

### 3. Heatmap is Uniform/Not Informative

Try:
- Different target layers (deeper or shallower)
- Different target classes
- Verify model is actually trained (not random weights)
- Check if input preprocessing matches training

### 4. Mismatched Shapes

Ensure input shape matches training:
```python
print(f"Input shape: {input_tensor.shape}")  # Should be [B, C, D, H, W]
# If wrong, check your preprocessing transforms
```

## Advanced Usage

### Multi-Class Analysis

```python
# Generate heatmaps for all classes
num_classes = 3
heatmaps = {}

for class_idx in range(num_classes):
    heatmaps[f'class_{class_idx}'] = gradcam.generate_cam(
        input_tensor, 
        target_class=class_idx
    )

# Compare classes
fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
for i, (name, hmap) in enumerate(heatmaps.items()):
    axes[i].imshow(hmap[hmap.shape[0]//2], cmap='jet')
    axes[i].set_title(name)
plt.show()
```

### Batch Processing

```python
import glob

# Process all images in a directory
image_paths = glob.glob('data/*.nii.gz')

for img_path in image_paths:
    # Load image
    data = transforms({"image": img_path})
    input_tensor = data["image"].unsqueeze(0).to(device)
    
    # Generate heatmap
    heatmap = gradcam.generate_cam(input_tensor, target_class=1)
    
    # Save
    output_name = Path(img_path).stem + '_heatmap.npy'
    np.save(output_name, heatmap)
```

### Custom Model Architectures

For non-MONAI models:

```python
from attribution_3d_heatmaps import GradCAM3D

# Works with any PyTorch model
custom_model = YourCustom3DModel()
custom_model.load_state_dict(torch.load('weights.pth'))
custom_model.eval()

# Just specify the target layer name
gradcam = GradCAM3D(custom_model, target_layer='encoder.layer4')
```

## File Structure

```
.
├── attribution_3d_heatmaps.py    # Main implementation
├── attribution_analysis.ipynb     # Interactive notebook
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{your_paper,
  title={3D Attribution Analysis for Medical Imaging},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

Also cite the original methods:
- GradCAM: Selvaraju et al., ICCV 2017
- GradCAM++: Chattopadhay et al., WACV 2018
- MONAI: Cardoso et al., 2022

## License

MIT License - feel free to use in your research and applications.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check MONAI documentation: https://docs.monai.io/
- Review GradCAM paper for methodology details

## Tips for Best Results

1. **Always verify** your model is loaded correctly and making reasonable predictions
2. **Use multiple methods** - different methods can reveal different insights
3. **Try different target layers** - deeper layers capture high-level features
4. **Normalize your inputs** the same way as during training
5. **Start with small volumes** for Occlusion Sensitivity, then scale up
6. **Save intermediate results** - computation can be expensive
7. **Document your parameters** - reproducibility is key

Happy analyzing! 🔍🧠
