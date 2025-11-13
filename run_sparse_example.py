"""
Simple runner for the CIFAR-10 sparse pixel example.

Run this script to demonstrate sparse pixel image-to-image learning.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from examples.sparse_pixel_example import main
    
    if __name__ == "__main__":
        print("Running CIFAR-10 sparse pixel image-to-image example...")
        print("Note: This requires PyTorch. Install with:")
        print("pip install -e .[examples]")
        print()
        print("This example demonstrates:")
        print("• Image-to-image learning with sparse pixel supervision")
        print("• U-Net CNN for pixel-level reconstruction")
        print("• Realistic federated learning scenarios")
        print("• Sparse input pixels (missing data) + sparse output labels")
        print()
        
        main()
        
except ImportError as e:
    print(f"ImportError: {e}")
    print()
    print("To run this example, install the required dependencies:")
    print("pip install -e .[examples]")
    print()
    print("This will install: torch, torchvision, matplotlib")