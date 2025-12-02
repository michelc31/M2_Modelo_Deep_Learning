import torch
import numpy as np
import pickle
from models import SimpleLSTM, CNNLSTM
from dataset import UCF101SkeletonDataset, load_data
import random

# Config
PKL_PATH = 'ucf101_2d.pkl'
# Must match training classes
TARGET_CLASSES = ['ApplyEyeMakeup', 'Archery', 'BoxingPunchingBag', 'Diving', 'Lunges']
FALLBACK_CLASSES = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam']
MAX_FRAMES = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_sample(model, sample, class_names):
    model.eval()
    with torch.no_grad():
        input_tensor = sample['data'].unsqueeze(0).to(DEVICE) # Add batch dim
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
        pred_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()
        
    return pred_class, confidence

def main():
    print("Loading data for prediction demo...")
    # Re-load data structure to get consistent mapping
    try:
        _, val_data, label_map, class_names = load_data(PKL_PATH, TARGET_CLASSES)
        if not val_data:
             _, val_data, label_map, class_names = load_data(PKL_PATH, FALLBACK_CLASSES)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_classes = len(class_names)
    
    # Initialize Dataset
    val_dataset = UCF101SkeletonDataset(val_data, label_map, max_frames=MAX_FRAMES)
    
    # Load Model
    # We'll try to load the improved model, assuming it was trained
    sample_input = val_dataset[0]['data']
    input_dim = sample_input.shape[1]
    
    model = CNNLSTM(input_dim, hidden_dim=128, num_classes=num_classes)
    model_path = 'improved_cnn_lstm.pth'
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Model file not found. Please run training first. Initializing random model for demo.")
    
    model.to(DEVICE)
    
    # Pick 5 random samples
    print("\n--- Running Predictions on Random Validation Samples ---")
    indices = random.sample(range(len(val_dataset)), 5)
    
    for idx in indices:
        sample = val_dataset[idx]
        true_label_idx = sample['label'].item()
        true_label = class_names[true_label_idx]
        
        pred_label, conf = predict_sample(model, sample, class_names)
        
        status = "CORRECT" if pred_label == true_label else "WRONG"
        print(f"Sample {idx}: True: {true_label:<20} Pred: {pred_label:<20} Conf: {conf:.4f} [{status}]")

if __name__ == "__main__":
    main()

