import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam

from model import VOIDataset, VOIClassifier, VOITrainer, prepare_data, prepare_dataset

def train(num_epochs=50, batch_size=32, learning_rate=0.001):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or prepare dataset
    if os.path.exists(os.path.join('Dataset', 'images.npy')):
        images = np.load(os.path.join('Dataset', 'images.npy'))
        labels = np.load(os.path.join('Dataset', 'labels.npy'))
    else:
        images, labels = prepare_dataset()

    # Prepare data splits and create datasets
    (train_x, train_y), (val_x, val_y), (test_x, test_y), class_weights = prepare_data(images, labels)
    
    train_dataset = VOIDataset(train_x, train_y, is_train=True)
    val_dataset = VOIDataset(val_x, val_y, is_train=False)
    test_dataset = VOIDataset(test_x, test_y, is_train=False)

    print("Datasets created")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Data loaders created")

    # Initialize model, optimizer, and loss function
    model = VOIClassifier().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    class_weights = class_weights.to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    print("Model initalized")

    # Initialize trainer
    trainer = VOITrainer(model, device)

    print("Trainer initalized")

    # Training loop
    best_auc = 0.0
    for epoch in range(num_epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, loss_function)
        
        # Validate
        auc, accuracy = trainer.validate(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with AUC: {best_auc:.4f}")
        
        print("-" * 40)

    # Final evaluation on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_auc, test_accuracy = trainer.validate(test_loader)
    print(f"\nFinal Test Results:")
    print(f"AUC: {test_auc:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    train() 