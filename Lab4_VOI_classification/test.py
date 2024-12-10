import torch
from model import test_model

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define test files and reference boundaries
    test_files = [
        "path/to/test/image1.nii.gz",
        "path/to/test/image2.nii.gz",
        # Add more test files as needed
    ]
    
    reference_boundaries = {
        "path/to/test/image1.nii.gz": (105, 137),
        "path/to/test/image2.nii.gz": (98, 142),
        # Add corresponding reference boundaries
    }
    
    # Run test
    avg_delta = test_model(
        model_path="best_metric_model.pth",
        test_files=test_files,
        reference_boundaries=reference_boundaries,
        device=device
    )
    
    return avg_delta

if __name__ == "__main__":
    test() 