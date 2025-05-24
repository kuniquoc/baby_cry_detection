import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from models import MobileNetV2_Crying
from utils import DatasetLoader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, writer, checkpoint_dir):
    """
    Train the model and validate
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 10
    
    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Binary classification: threshold at 0.5
            predicted = (outputs > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Store predictions and targets for metric calculation
            train_preds.extend(predicted.cpu().detach().numpy())
            train_targets.extend(targets.cpu().detach().numpy())
            
            pbar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'acc': 100.*train_correct/train_total
            })
            
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                # Binary classification: threshold at 0.5
                predicted = (outputs > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # Store predictions and targets for metric calculation
                val_preds.extend(predicted.cpu().detach().numpy())
                val_targets.extend(targets.cpu().detach().numpy())
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Calculate additional metrics
        train_precision = precision_score(train_targets, train_preds, zero_division=0)
        train_recall = recall_score(train_targets, train_preds, zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        
        val_precision = precision_score(val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step(avg_val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Precision/train', train_precision, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/train', train_recall, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('F1-Score/train', train_f1, epoch)
        writer.add_scalar('F1-Score/val', val_f1, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'best_val_loss': best_val_loss,
            'learning_rate': current_lr
        }
        
        # Save last checkpoint
        torch.save(checkpoint, checkpoint_dir / 'last_model.pth')
        
        # Save best model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f'\nNew best model saved! (Val Loss: {avg_val_loss:.4f})')
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f'\nValidation loss did not improve for {epochs_no_improve} epochs')
            
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        
        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs! No improvement for {early_stop_patience} consecutive epochs.')
            break
    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(checkpoint_dir / 'loss_plot.png')
    
    # Plot the learning rate changes
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(checkpoint_dir / 'lr_plot.png')
    
    return train_losses, val_losses

def main():
    # Configuration
    # Lấy thư mục gốc của project
    base_dir = Path(__file__).resolve().parent.parent  # Lấy thư mục cha của thư mục chứa file hiện tại
    data_dir = base_dir / 'data/dataset'
    processed_dir = base_dir / 'data/processed'
    runs_dir = base_dir / 'runs'
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    
    # Check GPU availability with better error handling
    try:
        if torch.cuda.is_available():
            # Set device to first GPU and test it
            device = torch.device('cuda:0')
            # Create a small tensor to test CUDA initialization
            test_tensor = torch.zeros(1, device=device)
            # If we get here, CUDA is working
            num_gpus = torch.cuda.device_count()
            print(f"Using {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        else:
            device = torch.device('cpu')
            print("CUDA not available, using CPU")
    except RuntimeError as e:
        print(f"CUDA error occurred: {e}")
        print("Falling back to CPU")
        device = torch.device('cpu')
        # Reset CUDA to prevent further errors
        torch.cuda.empty_cache()
    
    # Create directories for logs and checkpoints
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = runs_dir / timestamp
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(run_dir)
    
    # Load dataset with processed features support
    dataset_loader = DatasetLoader(data_dir=data_dir, processed_dir=processed_dir)
    train_loader, val_loader, test_loader = dataset_loader.prepare_dataset(
        batch_size=batch_size
    )
    
    # Get class weight for imbalanced dataset
    class_weights = dataset_loader.get_class_weights()
    pos_weight = class_weights.to(device) if torch.is_tensor(class_weights) else class_weights
    
    # Initialize model
    model = MobileNetV2_Crying()
    
    # Wrap model with DataParallel only if CUDA is available and we have multiple GPUs
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Move model to device
    model = model.to(device)
    
    # When using BCEWithLogitsLoss, the model should NOT apply sigmoid to outputs
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                 verbose=True, threshold=0.0001, min_lr=1e-6)
    
    # Train model
    try:
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            writer=writer,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save the loss data for future reference
        loss_data = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        torch.save(loss_data, checkpoint_dir / 'loss_data.pth')
    except Exception as e:
        print(f"Error during training: {e}")
        if device.type == 'cuda':
            print("Error might be CUDA-related. Try setting environment variable CUDA_LAUNCH_BLOCKING=1 for debugging")
            # Try to clean up CUDA memory
            torch.cuda.empty_cache()
    finally:
        writer.close()

if __name__ == '__main__':
    main()