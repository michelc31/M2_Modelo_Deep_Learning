import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import os
from tqdm import tqdm

from dataset import load_data, UCF101SkeletonDataset
from models import SimpleLSTM, CNNLSTM

# Config
PKL_PATH = 'ucf101_2d.pkl'
TARGET_CLASSES = ['ApplyEyeMakeup', 'Archery', 'BoxingPunchingBag', 'Diving', 'Lunges']
# Alternative fallback if those don't exist/match
FALLBACK_CLASSES = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam']

BATCH_SIZE = 32
MAX_FRAMES = 150
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        inputs = batch['data'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        # lengths = batch['length'] # Not used in simple padding implementation
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch['data'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(loader), accuracy, all_labels, all_preds

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

def run_experiment(model_class, model_name, train_loader, val_loader, input_dim, num_classes, class_names):
    print(f"\n--- Starting Experiment: {model_name} ---")
    model = model_class(input_dim, hidden_dim=128, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    # Final Evaluation
    _, final_acc, y_true, y_pred = evaluate(model, val_loader, criterion)
    print(f"\nFinal Validation Accuracy for {model_name}: {final_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    plot_confusion_matrix(y_true, y_pred, class_names, f'Confusion Matrix {model_name}')
    
    return history, model

def main():
    print(f"Device: {DEVICE}")
    
    # 1. Load Data
    print("Loading data...")
    # First try primary target classes
    try:
        train_data, val_data, label_map, class_names = load_data(PKL_PATH, TARGET_CLASSES)
        # Check if we actually got data matching those classes
        if not train_data:
            print("Primary classes not found/matched. Trying fallback.")
            train_data, val_data, label_map, class_names = load_data(PKL_PATH, FALLBACK_CLASSES)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    if len(train_data) == 0:
        print("No training data found. Exiting.")
        return

    # Create Datasets
    train_dataset = UCF101SkeletonDataset(train_data, label_map, max_frames=MAX_FRAMES)
    val_dataset = UCF101SkeletonDataset(val_data, label_map, max_frames=MAX_FRAMES)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Determine input dim
    # Take one sample to check shape
    sample = train_dataset[0]['data']
    input_dim = sample.shape[1] # (V*C)
    print(f"Input feature dimension: {input_dim}")
    
    # 2. Run Baseline (LSTM)
    lstm_history, lstm_model = run_experiment(SimpleLSTM, "Baseline LSTM", train_loader, val_loader, input_dim, len(class_names), class_names)
    
    # 3. Run Improved (CNN-LSTM)
    cnn_history, cnn_model = run_experiment(CNNLSTM, "Improved CNN-LSTM", train_loader, val_loader, input_dim, len(class_names), class_names)
    
    # 4. Compare Plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history['val_acc'], label='Baseline LSTM')
    plt.plot(cnn_history['val_acc'], label='Improved CNN-LSTM')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history['val_loss'], label='Baseline LSTM')
    plt.plot(cnn_history['val_loss'], label='Improved CNN-LSTM')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    print("\nPlots saved to comparison_plot.png and confusion matrices.")
    
    # Save models
    torch.save(lstm_model.state_dict(), 'baseline_lstm.pth')
    torch.save(cnn_model.state_dict(), 'improved_cnn_lstm.pth')

if __name__ == "__main__":
    main()

