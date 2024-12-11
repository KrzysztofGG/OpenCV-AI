import os
import nibabel as nib
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F


class VOIDataset(Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        
        # Define transforms for training and validation
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Normalize(mean=[0.456], std=[0.224])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.456], std=[0.224])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Images are already tensors from prepare_data
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class VOIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        # We'll calculate fc1 input size in forward pass
        self.fc1 = None
        self.fc2 = nn.Linear(512, 2)
        
    def _get_conv_output_size(self, shape):
        # Helper function to calculate output size
        # Create tensor on the same device as the model's parameters
        device = next(self.parameters()).device
        x = torch.zeros(1, 1, *shape).to(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel() // x.shape[0]
        
    def forward(self, x):
        # Initialize fc1 if it's the first forward pass
        if self.fc1 is None:
            input_size = self._get_conv_output_size(x.shape[2:])
            self.fc1 = nn.Linear(input_size, 512).to(x.device)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VOITrainer:
    def __init__(self, model, device, num_classes=2):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
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
            correct = 0
            total = 0
            y_true = []
            y_scores = []
            
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(self.device), val_data[1].to(self.device)
                outputs = self.model(val_images)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
                
                # Store predictions for AUC calculation
                y_true.extend(val_labels.cpu().numpy())
                y_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            
            accuracy = correct / total
            
            # Calculate AUC using sklearn
            from sklearn.metrics import roc_auc_score
            auc_result = roc_auc_score(y_true, y_scores)
            
            return auc_result, accuracy

def prepare_data(images_array, labels_array, train_size=0.7, val_size=0.15):
    # Split indices
    indices = np.random.permutation(len(images_array))
    train_end = int(train_size * len(indices))
    val_end = int((train_size + val_size) * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create data splits - already as tensors with correct shape
    train_x = torch.tensor(images_array[train_indices]).unsqueeze(1).float()  # Add channel dimension
    train_y = torch.tensor(labels_array[train_indices]).long()
    
    val_x = torch.tensor(images_array[val_indices]).unsqueeze(1).float()
    val_y = torch.tensor(labels_array[val_indices]).long()
    
    test_x = torch.tensor(images_array[test_indices]).unsqueeze(1).float()
    test_y = torch.tensor(labels_array[test_indices]).long()
    
    # Calculate class weights
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
    # Initialize model
    model = VOIClassifier().to(device)
    
    # Create a dummy input to initialize fc1 layer
    dummy_input = torch.zeros(1, 1, 256, 256).to(device)  # Assuming 256x256 input size
    _ = model(dummy_input)  # This will initialize fc1
    
    # Now load the state dict
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

# Configuration
DATA_ROOT = 'Data'  # Root directory containing .nii.gz files
DATASET_ROOT = 'Dataset'  # Directory for processed numpy arrays
MARKINGS_FILE = os.path.join(DATA_ROOT, 'oznaczenia.txt')

def prepare_dataset():
    """Prepare and save the dataset from NIFTI files"""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    
    # Read markings file
    with open(MARKINGS_FILE, 'r') as file:
        lines = file.readlines()

    images = []
    labels = []
    target_size = (256, 256)  # Define target size for resizing
    
    # Process each NIFTI file
    for line in lines:
        if not line.strip():  # Skip empty lines
            continue
        print(line)
        file_name, index1, index0 = line.strip().split('\t')
        img_path = os.path.join(DATA_ROOT, file_name)
        
        # Load NIFTI file
        ct_img = nib.load(img_path)
        ct_data = ct_img.get_fdata()
        
        # Process each slice
        for i in range(ct_data.shape[2]):
            # Label is 1 if slice is within VOI range
            label = 1 if i in range(int(index0), int(index1)+1) else 0
            slice_data = ct_data[:, :, i]
            
            # Resize the slice
            resized_slice = resize(slice_data, target_size, mode='reflect', preserve_range=True)
            
            # Normalize slice
            normalized_slice = normalize_slice(resized_slice)
            
            images.append(normalized_slice)
            labels.append(label)

    print(images[0].shape)
    print(len(images))
    print(labels[0])

    # Convert to numpy arrays
    images_array = np.array(images).astype(np.float32)
    labels_array = np.array(labels).astype(np.int64)

    print(images_array.shape)
    print(labels_array.shape)
    
    # Save processed data
    np.save(os.path.join(DATASET_ROOT, 'images.npy'), images_array)
    np.save(os.path.join(DATASET_ROOT, 'labels.npy'), labels_array)
    
    return images_array, labels_array
