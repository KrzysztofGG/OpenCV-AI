import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from monai.networks.nets import DenseNet121

from model import VOIDataset, VOITrainer, prepare_data

def train():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    batch_size = 32
    learning_rate = 1e-5
    max_epochs = 4
    
    # Load data
    images_array = np.load(os.path.join('Dataset', 'images.npy'))
    labels_array = np.load(os.path.join('Dataset', 'labels.npy'))
    
    # Prepare datasets
    (train_x, train_y), (val_x, val_y), (test_x, test_y), class_weights = prepare_data(
        images_array, labels_array
    )
    
    train_ds = VOIDataset(train_x, train_y)
    val_ds = VOIDataset(val_x, val_y)
    test_ds = VOIDataset(test_x, test_y)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    # Initialize model and trainer
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_classes)
    trainer = VOITrainer(model, device, num_classes)
    
    # Training setup
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        
        # Train
        epoch_loss = trainer.train_epoch(train_loader, optimizer, loss_function)
        print(f"Average loss: {epoch_loss:.4f}")
        
        # Validate
        auc_result, accuracy = trainer.validate(val_loader)
        
        # Save best model
        if auc_result > best_metric:
            best_metric = auc_result
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model.pth")
            print("Saved new best metric model")
            
        print(f"AUC: {auc_result:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Best AUC: {best_metric:.4f} at epoch: {best_metric_epoch}")

    return "best_metric_model.pth"

if __name__ == "__main__":
    train() 