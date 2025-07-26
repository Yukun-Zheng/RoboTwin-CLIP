#!/usr/bin/env python3
"""
Example usage script for Image Segmentation and CLIP Analysis Tool

This script demonstrates how to use the image_segmentation_clip_analysis.py tool
with different configurations and custom descriptions.
"""

import os
import sys
from image_segmentation_clip_analysis import ImageSegmentationCLIPAnalyzer

def example_bottle_analysis():
    """
    Example analysis for a bottle image
    """
    print("=" * 60)
    print("EXAMPLE 1: Bottle Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ImageSegmentationCLIPAnalyzer()
    
    # Custom descriptions for bottle analysis
    bottle_descriptions = [
        "a water bottle",
        "a plastic bottle", 
        "a glass bottle",
        "a drink bottle",
        "a shoe",
        "a cup",
        "a mug",
        "a container",
        "a beverage bottle",
        "a liquid container"
    ]
    
    # Replace with your actual image path
    image_path = "bottle_image.jpg"  # Put your bottle image here
    
    if os.path.exists(image_path):
        results = analyzer.analyze_image(
            image_path=image_path,
            descriptions=bottle_descriptions,
            use_sam=True,
            sam_model_type="vit_b",
            save_results=True
        )
        
        print(f"\nAnalysis complete for {image_path}")
        print(f"Best match: {results['best_match']} (score: {results['best_score']:.3f})")
    else:
        print(f"Image not found: {image_path}")
        print("Please place your bottle image in the current directory and update the path.")

def example_shoe_analysis():
    """
    Example analysis for a shoe image
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Shoe Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ImageSegmentationCLIPAnalyzer()
    
    # Custom descriptions for shoe analysis
    shoe_descriptions = [
        "a shoe",
        "a sneaker",
        "a running shoe",
        "a boot",
        "a leather shoe",
        "a sports shoe",
        "a water bottle",
        "a cup",
        "a bag",
        "footwear"
    ]
    
    # Replace with your actual image path
    image_path = "shoe_image.jpg"  # Put your shoe image here
    
    if os.path.exists(image_path):
        results = analyzer.analyze_image(
            image_path=image_path,
            descriptions=shoe_descriptions,
            use_sam=True,
            sam_model_type="vit_b",
            save_results=True
        )
        
        print(f"\nAnalysis complete for {image_path}")
        print(f"Best match: {results['best_match']} (score: {results['best_score']:.3f})")
    else:
        print(f"Image not found: {image_path}")
        print("Please place your shoe image in the current directory and update the path.")

def example_custom_analysis():
    """
    Example with completely custom descriptions
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ImageSegmentationCLIPAnalyzer()
    
    # Very specific custom descriptions
    custom_descriptions = [
        "a red water bottle",
        "a blue water bottle", 
        "a clear plastic bottle",
        "a metal water bottle",
        "a white sneaker",
        "a black shoe",
        "a brown leather shoe",
        "a colorful running shoe"
    ]
    
    # Replace with your actual image path
    image_path = "your_image.jpg"  # Put your image here
    
    if os.path.exists(image_path):
        results = analyzer.analyze_image(
            image_path=image_path,
            descriptions=custom_descriptions,
            use_sam=True,
            sam_model_type="vit_l",
            save_results=True
        )
        
        print(f"\nAnalysis complete for {image_path}")
        print(f"Best match: {results['best_match']} (score: {results['best_score']:.3f})")
    else:
        print(f"Image not found: {image_path}")
        print("Please place your image in the current directory and update the path.")

def quick_test_with_sample():
    """
    Quick test function - you can modify this for your specific image
    """
    print("\n" + "=" * 60)
    print("QUICK TEST - Modify this function for your image")
    print("=" * 60)
    
    # TODO: Replace with your actual image path
    your_image_path = "PUT_YOUR_IMAGE_PATH_HERE.jpg"
    
    if not os.path.exists(your_image_path):
        print(f"Please update 'your_image_path' variable with your actual image path.")
        print(f"Current path: {your_image_path}")
        return
    
    # Initialize analyzer
    analyzer = ImageSegmentationCLIPAnalyzer()
    
    # Default descriptions (good for bottles and shoes)
    results = analyzer.analyze_image(
        image_path=your_image_path,
        descriptions=None,  # Uses default descriptions
        use_sam=True,
        sam_model_type="vit_b",
        save_results=True
    )
    
    print(f"\nQuick test complete!")
    print(f"Best match: {results['best_match']} (score: {results['best_score']:.3f})")

def main():
    """
    Main function to run examples
    """
    print("Image Segmentation and CLIP Analysis - Example Usage")
    print("\nThis script shows how to use the analysis tool with different configurations.")
    print("\nIMPORTANT: Update the image paths in the functions below with your actual images!")
    
    # Run examples (comment out the ones you don't need)
    
    # Example 1: Bottle analysis
    # example_bottle_analysis()
    
    # Example 2: Shoe analysis  
    # example_shoe_analysis()
    
    # Example 3: Custom descriptions
    # example_custom_analysis()
    
    # Quick test - modify this for your image
    quick_test_with_sample()
    
    print("\n" + "=" * 60)
    print("USAGE TIPS:")
    print("=" * 60)
    print("1. Place your images in the current directory")
    print("2. Update the image paths in the example functions")
    print("3. Uncomment the example you want to run")
    print("4. Run: python example_usage.py")
    print("\nAlternatively, use the command line tool directly:")
    print("python image_segmentation_clip_analysis.py --image your_image.jpg")
    print("\nFor more options:")
    print("python image_segmentation_clip_analysis.py --help")

if __name__ == "__main__":
    main()