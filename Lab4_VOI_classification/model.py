import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from monai.apps import download_and_extract
from monai.config import print_config
# from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

class VOIDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = Compose([
            ScaleIntensity(),
            RandFlip(prob=0.5),
            RandRotate(range_x=15, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)
        ]) if transform is None else transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class VOITrainer:
    def __init__(self, model, device, num_classes=2):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        self.y_pred_trans = Compose([Activations(softmax=True)])
        self.y_trans = Compose([AsDiscrete(to_onehot=num_classes)])
        self.auc_metric = ROCAUCMetric()
        
    def train_epoch(self, train_loader, optimizer, loss_function):
        self.model.train()
        epoch_loss = 0
        steps = 0
        
        for batch_data in train_loader:
            steps += 1
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / steps
    
    def validate(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=self.device)
            y = torch.tensor([], dtype=torch.long, device=self.device)
            
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)
                y_pred = torch.cat([y_pred, self.model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            
            y_onehot = self.y_trans(y)
            y_pred_act = self.y_pred_trans(y_pred)
            self.auc_metric(y_pred_act, y_onehot)
            auc_result = self.auc_metric.aggregate()
            self.auc_metric.reset()
            
            accuracy = torch.eq(y_pred.argmax(dim=1), y).sum().item() / len(y)
            
            return auc_result, accuracy

def prepare_data(images_array, labels_array, train_size=0.7, val_size=0.15):
    # Split indices
    indices = np.random.permutation(len(images_array))
    train_end = int(train_size * len(indices))
    val_end = int((train_size + val_size) * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create data splits
    train_x = [torch.tensor(images_array[i]).unsqueeze(0) for i in train_indices]
    train_y = labels_array[train_indices]
    
    val_x = [torch.tensor(images_array[i]).unsqueeze(0) for i in val_indices]
    val_y = labels_array[val_indices]
    
    test_x = [torch.tensor(images_array[i]).unsqueeze(0) for i in test_indices]
    test_y = labels_array[test_indices]
    
    # Add class weights calculation
    class_counts = np.bincount(labels_array)
    total_samples = len(labels_array)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) 
                                     for count in class_counts])
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y), class_weights

def find_voi_boundaries(predictions, window_size=5, threshold=0.6):
    """
    Find VOI boundaries from sequence of predictions using sliding window
    """
    # Smooth predictions using sliding window
    smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')
    # Find transitions above threshold
    transitions = np.where(np.diff((smoothed > threshold).astype(int)))[0]
    
    if len(transitions) >= 2:
        start_idx = transitions[0]
        end_idx = transitions[-1]
        return start_idx, end_idx
    return None, None

def load_nifti_file(file_path):
    """Load and preprocess NIFTI file"""
    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()
    # Normalize and prepare slices
    slices = [normalize_slice(image_data[:, :, i]) for i in range(image_data.shape[2])]
    return np.array(slices)

def majority_vote_predictions(model_predictions, threshold=3):
    """Combine predictions from multiple models"""
    stacked_preds = np.stack(model_predictions)
    return (stacked_preds.sum(axis=0) >= threshold).astype(int)

def normalize_slice(slice_data, min_bound=-1000, max_bound=400, output_range=(0, 1)):
    """
    Normalize CT slice data to a fixed range.
    
    Args:
        slice_data: 2D numpy array of CT slice
        min_bound: minimum HU value to consider (typically -1000 for air)
        max_bound: maximum HU value to consider (typically 400 for soft tissue)
        output_range: tuple of (min, max) values for output range
    
    Returns:
        Normalized 2D numpy array
    """
    # Clip HU values to specified range
    slice_data = np.clip(slice_data, min_bound, max_bound)
    
    # Normalize to [0, 1] range
    slice_data = (slice_data - min_bound) / (max_bound - min_bound)
    
    # Scale to desired output range
    if output_range != (0, 1):
        slice_data = slice_data * (output_range[1] - output_range[0]) + output_range[0]
    
    return slice_data

def load_and_preprocess_test_image(file_path, target_size=(256, 256)):
    """
    Load and preprocess a single test image.
    Returns preprocessed slices ready for model inference.
    """
    nifti_img = nib.load(file_path)
    image_data = nifti_img.get_fdata()
    
    # Preprocess each slice
    processed_slices = []
    for i in range(image_data.shape[2]):
        slice_data = image_data[:, :, i]
        # Resize
        resized_slice = resize(slice_data, target_size, mode='reflect', preserve_range=True)
        # Normalize
        normalized_slice = normalize_slice(resized_slice)
        processed_slices.append(normalized_slice)
    
    return np.array(processed_slices)

def predict_voi_boundaries(model, image_path, device, window_size=7, threshold=0.5):
    """
    Predict VOI boundaries for a single test image.
    Returns predicted start and end positions.
    """
    # Load and preprocess image
    slices = load_and_preprocess_test_image(image_path)
    
    # Prepare for model
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for slice_data in slices:
            # Prepare input
            input_tensor = torch.tensor(slice_data).unsqueeze(0).unsqueeze(0).float().to(device)
            # Get prediction
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            predictions.append(prob)
    
    # Smooth predictions
    predictions = np.array(predictions)
    smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')
    
    # Find transitions
    binary_preds = (smoothed > threshold).astype(int)
    transitions = np.where(np.diff(binary_preds))[0]
    
    if len(transitions) >= 2:
        # Add offset due to smoothing window
        start_idx = transitions[0] + window_size//2
        end_idx = transitions[-1] + window_size//2
        return start_idx, end_idx
    
    return None, None

def evaluate_predictions(predicted_boundaries, reference_boundaries):
    """
    Calculate delta metric between predicted and reference boundaries.
    """
    pred_start, pred_end = predicted_boundaries
    ref_start, ref_end = reference_boundaries
    
    if pred_start is None or pred_end is None:
        return float('inf')  # or some large penalty value
    
    delta = abs(pred_start - ref_start) + abs(pred_end - ref_end)
    return delta

def test_model(model_path, test_files, reference_boundaries, device):
    """
    Test model on multiple files and calculate average delta.
    
    Args:
        model_path: Path to saved model
        test_files: List of test image paths
        reference_boundaries: Dictionary mapping file paths to (start, end) tuples
        device: torch device
    
    Returns:
        Average delta across all test files
    """
    # Load model
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    deltas = []
    for file_path in test_files:
        # Predict boundaries
        pred_start, pred_end = predict_voi_boundaries(model, file_path, device)
        
        # Get reference boundaries
        ref_bounds = reference_boundaries[file_path]
        
        # Calculate delta
        delta = evaluate_predictions((pred_start, pred_end), ref_bounds)
        deltas.append(delta)
        
        print(f"File: {file_path}")
        print(f"Predicted: {pred_start}-{pred_end}")
        print(f"Reference: {ref_bounds[0]}-{ref_bounds[1]}")
        print(f"Delta: {delta}\n")
    
    avg_delta = np.mean(deltas)
    print(f"Average delta: {avg_delta}")
    return avg_delta
