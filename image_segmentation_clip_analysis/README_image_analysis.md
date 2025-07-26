# Image Segmentation and CLIP Analysis Tool

This tool performs image segmentation to extract objects (like bottles or shoes) from photos and uses CLIP to analyze text-image similarity with heatmap visualizations.

## Features

- **Image Segmentation**: Extract objects using GrabCut, Watershed, or Threshold methods
- **CLIP Text Encoding**: Encode multiple text descriptions using CLIP's text encoder
- **Similarity Analysis**: Calculate cosine similarity between image and text features
- **Heatmap Visualization**: Generate beautiful heatmaps showing similarity scores
- **Comprehensive Output**: Save extracted objects, masks, and analysis results

## Installation

### Prerequisites

Make sure you have the required dependencies:

```bash
pip install torch torchvision
pip install matplotlib seaborn
pip install opencv-python
pip install Pillow
pip install numpy
pip install ftfy regex tqdm  # For CLIP

# Install SAM (Segment Anything Model) for advanced segmentation
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### SAM Model Checkpoints

Download the SAM model checkpoints you want to use:

```bash
# Download SAM checkpoints (choose one or more)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  # Base model (358MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  # Large model (1.2GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # Huge model (2.4GB)
```

### Setup

1. Ensure CLIP is available in your Python path (already configured in the script)
2. Place your images in the current directory
3. Run the analysis tool

## Usage

### Method 1: Command Line Interface

```bash
# Basic usage with SAM
python image_segmentation_clip_analysis.py --image your_bottle.jpg

# Custom text descriptions
python image_segmentation_clip_analysis.py --image your_shoe.jpg --descriptions "a running shoe" "a leather boot" "a water bottle"

# With different SAM model
python image_segmentation_clip_analysis.py --image your_image.jpg --sam-model vit_l

# With custom SAM checkpoint path
python image_segmentation_clip_analysis.py --image your_image.jpg --sam-checkpoint /path/to/sam_vit_b_01ec64.pth

# Without SAM (use fallback segmentation)
python image_segmentation_clip_analysis.py --image your_image.jpg --no-sam

# Don't save results (just display)
python image_segmentation_clip_analysis.py --image your_image.jpg --no-save

# Use different CLIP model
python image_segmentation_clip_analysis.py --image your_image.jpg --clip-model "ViT-B/16"
```

### Method 2: Python Script

```python
from image_segmentation_clip_analysis import ImageSegmentationCLIPAnalyzer

# Initialize analyzer
analyzer = ImageSegmentationCLIPAnalyzer()

# Custom descriptions for your analysis
descriptions = [
    "a water bottle",
    "a plastic bottle", 
    "a shoe",
    "a sneaker",
    "a cup"
]

# Run analysis with SAM
results = analyzer.analyze_image(
    image_path="your_image.jpg",
    descriptions=descriptions,
    use_sam=True,
    sam_model_type="vit_b",
    save_results=True
)

print(f"Best match: {results['best_match']}")
print(f"Similarity score: {results['best_score']:.3f}")

# Use without SAM (fallback method)
results_fallback = analyzer.analyze_image(
    image_path="your_image.jpg",
    use_sam=False,
    save_results=True
)
```

### Method 3: Example Script

```bash
# Run the example script (modify paths first)
python example_usage.py
```

## Segmentation Methods

The tool supports advanced segmentation using SAM (Segment Anything Model):

### SAM Models
1. **vit_b** (default): Base ViT model - fastest, good quality
2. **vit_l**: Large ViT model - better quality, slower
3. **vit_h**: Huge ViT model - best quality, slowest

### Fallback Method
When SAM is not available or disabled:
- **GrabCut**: Interactive foreground extraction using OpenCV

### SAM Features
- Automatic mask generation
- Intelligent mask selection based on area and centrality
- High-quality segmentation for various object types
- GPU acceleration support

## Default Text Descriptions

The tool includes these default descriptions:
- "a water bottle"
- "a shoe"
- "a cup"
- "a mug"
- "a sneaker"
- "a boot"
- "a plastic bottle"
- "a glass bottle"
- "a running shoe"
- "a leather shoe"

## Output Files

When `save_results=True`, the tool generates:

1. `{image_name}_comprehensive_analysis.png`: Complete visualization with all steps
2. `{image_name}_similarity_heatmap.png`: Detailed heatmap of similarities
3. `{image_name}_extracted_object.png`: Segmented object with background removed

## Example Results

The tool will:
1. Load and segment your image
2. Extract the main object (bottle, shoe, etc.)
3. Encode your text descriptions using CLIP
4. Calculate similarity scores
5. Generate visualizations showing:
   - Original image
   - Segmentation mask
   - Extracted object
   - Similarity scores as bar chart and heatmap

## Tips for Best Results

### Image Quality
- Use clear, well-lit photos
- Ensure the object is the main subject
- Place the main object in the center of the image for better SAM selection

### Text Descriptions
- Be specific ("red water bottle" vs "bottle")
- Include variations ("shoe", "sneaker", "footwear")
- Add negative examples for comparison

### SAM Model Selection
- Use `vit_b` for quick analysis
- Use `vit_l` or `vit_h` for better segmentation quality
- SAM works well with complex backgrounds, but simple ones are still preferred
- Use CUDA-enabled GPU for faster SAM processing

## Troubleshooting

### Common Issues

1. **"Could not load image"**: Check file path and format (JPG, PNG supported)
2. **SAM not available**: Install SAM using the provided pip command
3. **SAM checkpoint not found**: Download the required checkpoint files
4. **Poor segmentation**: Try different SAM models or use fallback method
5. **CUDA out of memory**: Use smaller SAM model or CPU instead of GPU
6. **Low similarity scores**: Try more specific text descriptions

### Performance

- First run may be slower due to CLIP model loading
- GPU acceleration available if CUDA is installed
- Larger images take longer to process

## Advanced Usage

### Custom CLIP Models

Supported CLIP models:
- `ViT-B/32` (default, fastest)
- `ViT-B/16` (better quality, slower)
- `ViT-L/14` (best quality, slowest)

### Integration with Other Tools

The analyzer can be easily integrated into larger pipelines:

```python
# Use in a batch processing pipeline with SAM
analyzer = ImageSegmentationCLIPAnalyzer()

for image_path in image_list:
    results = analyzer.analyze_image(
        image_path, 
        use_sam=True,
        sam_model_type="vit_b",  # Use faster model for batch processing
        save_results=False
    )
    # Process results...
```

## File Structure

```
.
├── image_segmentation_clip_analysis.py  # Main analysis tool
├── example_usage.py                     # Usage examples
├── README_image_analysis.md             # This file
├── your_image.jpg                       # Your input images
└── output_files/                        # Generated results
    ├── image_comprehensive_analysis.png
    ├── image_similarity_heatmap.png
    └── image_extracted_object.png
```

## License

This tool uses CLIP (MIT License) and OpenCV. Please ensure compliance with respective licenses.

---

**Happy analyzing! 🔍📸**