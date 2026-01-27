import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.segmentation import GlaucomaSegmentationModel

def test_segformer():
    print("Testing SegFormer Model...")
    try:
        model = GlaucomaSegmentationModel()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy input (Batch, Channels, Height, Width)
    # SegFormer typically resized to 512x512 internally or via processor, but let's test with a standard size
    dummy_input = torch.randn(1, 3, 512, 512)
    
    try:
        logits = model(dummy_input)
        print(f"Forward pass successful. Output shape: {logits.shape}")
        
        # Check output shape
        assert logits.shape == (1, 3, 512, 512), f"Expected (1, 3, 512, 512), got {logits.shape}"
        print("Output shape verification passed.")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_segformer()
