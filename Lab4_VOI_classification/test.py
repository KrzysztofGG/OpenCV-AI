import torch
import os
from model import test_model

def load_reference_boundaries(ranges_file, test_dir):
    """Load reference boundaries from ranges.txt file"""
    reference_boundaries = {}
    with open(ranges_file, 'r') as f:
        for i, line in enumerate(f):
            if line.strip():  # Skip empty lines
                start_idx, end_idx = map(int, line.strip().split())
                file_name = f"{i}.nii.gz"  # Assuming file naming convention
                reference_boundaries[os.path.join(test_dir, file_name)] = (start_idx, end_idx)
    return reference_boundaries

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get test files from testData directory
    test_dir = "testData"
    ranges_file = os.path.join(test_dir, "ranges.txt")
    
    # Load reference boundaries
    reference_boundaries = load_reference_boundaries(ranges_file, test_dir)
    
    # Get corresponding test files
    test_files = list(reference_boundaries.keys())
    
    if not test_files:
        raise ValueError("No test files found in testData directory or ranges.txt is empty")
    
    # Run test
    avg_delta = test_model(
        model_path="best_model.pth",
        test_files=test_files,
        reference_boundaries=reference_boundaries,
        device=device
    )
    
    return avg_delta

if __name__ == "__main__":
    avg_delta = test()
    with open('avg_delta.txt', 'w') as f:
        f.write(f"{avg_delta}")