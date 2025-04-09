import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from models import MobileNetV2_Crying
from utils import DatasetLoader

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, writer, checkpoint_dir):
    """
    Train the model and validate
    """
    best_val_loss = float('inf')
    
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
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
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
            'best_val_loss': best_val_loss
        }
        
        # Save last checkpoint
        torch.save(checkpoint, checkpoint_dir / 'last_model.pth')
        
        # Save best model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
            print(f'\nNew best model saved! (Val Loss: {avg_val_loss:.4f})')
            
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    pos_weight = dataset_loader.get_class_weights().to(device)
    
    # Initialize model
    model = MobileNetV2_Crying().to(device)
    
    # When using BCEWithLogitsLoss, the model should NOT apply sigmoid to outputs
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        writer=writer,
        checkpoint_dir=checkpoint_dir
    )
    
    writer.close()

if __name__ == '__main__':
    main()