import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class UCF101SkeletonDataset(Dataset):
    def __init__(self, annotations, label_map, max_frames=300, transform=None):
        """
        Args:
            annotations (list): List of annotation dicts.
            label_map (dict): Mapping from original label int to new class index (0 to C-1).
            max_frames (int): Max frames to keep (padding/truncating).
        """
        self.annotations = annotations
        self.label_map = label_map
        self.max_frames = max_frames
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Keypoint shape: (M, T, V, C)
        kps = item['keypoint']
        scores = item['keypoint_score'] # (M, T, V)
        
        # Select the person with the highest total score across all frames
        # Sum scores over T and V -> (M,)
        person_scores = scores.sum(axis=(1, 2))
        best_person_idx = np.argmax(person_scores)
        
        # Select that person: (T, V, C)
        kps_person = kps[best_person_idx] # T, V, C
        
        # Normalize by image shape
        h, w = item['img_shape']
        kps_person[..., 0] = kps_person[..., 0] / w
        kps_person[..., 1] = kps_person[..., 1] / h
        
        # Flatten V and C: (T, V*C)
        T, V, C = kps_person.shape
        kps_flat = kps_person.reshape(T, V * C)
        
        # Handle temporal dimension (Padding/Truncating)
        # RNNs can handle variable length but for batching in PyTorch without pack_padded_sequence
        # (which is better but more complex for this demo), we often pad.
        # Or we can use pack_padded_sequence. Let's do simple padding for now.
        
        if T > self.max_frames:
            # Truncate (maybe take middle?)
            # Let's take the beginning for simplicity, or uniform sampling
            # Uniform sampling is better for action recognition
            indices = np.linspace(0, T-1, self.max_frames).astype(int)
            kps_flat = kps_flat[indices]
            actual_length = self.max_frames
        else:
            actual_length = T
            # Padding will be handled by collate_fn or here
            # Let's pad here with zeros to max_frames
            pad_len = self.max_frames - T
            if pad_len > 0:
                padding = np.zeros((pad_len, V*C), dtype=np.float32)
                kps_flat = np.concatenate([kps_flat, padding], axis=0)
        
        # Label
        original_label = item['label']
        new_label = self.label_map[original_label]
        
        return {
            'data': torch.tensor(kps_flat, dtype=torch.float32),
            'label': torch.tensor(new_label, dtype=torch.long),
            'length': torch.tensor(actual_length, dtype=torch.long)
        }

def load_data(pkl_path, target_classes=None):
    """
    Load data and filter by target classes (list of strings).
    If target_classes is None, use all.
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at {pkl_path}")
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    annotations = data['annotations']
    splits = data['split'] # keys: 'train1', 'test1', etc.
    
    # Build label to class name mapping
    # We assume video names are like v_ClassName_gXX_cXX
    label_to_name = {}
    for item in annotations:
        if item['label'] not in label_to_name:
            vid_name = item['frame_dir']
            # UCF101 format: v_ApplyEyeMakeup_g08_c01
            parts = vid_name.split('_')
            if len(parts) > 1:
                class_name = parts[1]
                label_to_name[item['label']] = class_name
    
    # If target_classes is provided, filter
    if target_classes:
        # Find labels corresponding to target_classes
        target_labels = []
        for lbl, name in label_to_name.items():
            if name in target_classes:
                target_labels.append(lbl)
        
        if not target_labels:
            print("Warning: No target classes found in dataset. Using all classes.")
            filtered_annotations = annotations
            final_classes = list(label_to_name.values())
        else:
            filtered_annotations = [x for x in annotations if x['label'] in target_labels]
            final_classes = target_classes
            # Sort target labels to ensure consistent mapping
            target_labels.sort()
    else:
        filtered_annotations = annotations
        final_classes = list(set(label_to_name.values()))
        target_labels = sorted(list(label_to_name.keys()))
        
    # Create mapping from original label to 0..K-1
    new_label_map = {old_lbl: new_idx for new_idx, old_lbl in enumerate(target_labels)}
    
    # Split into train and val
    # We use 'train1' and 'test1' if available
    train_ids = set()
    val_ids = set()
    
    if 'train1' in splits and 'test1' in splits:
        train_ids = set(splits['train1'])
        val_ids = set(splits['test1'])
    else:
        # Fallback: random split
        print("Warning: Standard splits not found. Performing random split.")
        all_names = [x['frame_dir'] for x in filtered_annotations]
        # ... logic to split ...
        # For now let's assume the pickle from mmaction has 'train1'
        pass

    train_data = []
    val_data = []
    
    for item in filtered_annotations:
        vid_name = item['frame_dir']
        # Some pickle files might strip extensions or match exactly
        # Usually mmaction split lists are just video names without extension
        if vid_name in train_ids:
            train_data.append(item)
        elif vid_name in val_ids:
            val_data.append(item)
        else:
            # Try matching without extension if vid_name has it
            name_no_ext = os.path.splitext(vid_name)[0]
            if name_no_ext in train_ids:
                train_data.append(item)
            elif name_no_ext in val_ids:
                val_data.append(item)
    
    # If splits were empty (maybe names didn't match), do a manual random split
    if len(train_data) == 0:
        print("Manual split needed.")
        np.random.shuffle(filtered_annotations)
        split_idx = int(len(filtered_annotations) * 0.8)
        train_data = filtered_annotations[:split_idx]
        val_data = filtered_annotations[split_idx:]
        
    return train_data, val_data, new_label_map, final_classes


