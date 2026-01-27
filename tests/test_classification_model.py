import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.classification import GlaucomaClassifier

def test_swinv2():
    print("Testing SwinV2 Classification Model...")
    try:
        model = GlaucomaClassifier()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy input (Batch, Channels, Height, Width)
    # SwinV2 Tiny usually expects 256x256 or similar. Let's try 3x256x256
    dummy_input = torch.randn(1, 3, 256, 256)
    
    try:
        logits = model(dummy_input)
        print(f"Forward pass successful. Output shape: {logits.shape}")
        
        # Check output shape
        # Expecting (1, 2) roughly
        print(f"Logits: {logits}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test_swinv2()
