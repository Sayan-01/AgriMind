"""
Utility functions for training and evaluation.
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(
    state: Dict[str, Any], 
    is_best: bool, 
    checkpoint_path: str,
    best_path: str = None
):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    torch.save(state, checkpoint_path)
    
    if is_best and best_path:
        shutil.copy2(checkpoint_path, best_path)
        logger.info(f"New best model saved to {best_path}")


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    return checkpoint


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make sure to set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_class_weights(class_counts: Dict[str, int]) -> torch.Tensor:
    """Create class weights for handling imbalanced datasets."""
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = []
    for class_name in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_name])
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training history (loss and accuracy curves)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    if 'lr' in history:
        ax3.plot(history['lr'], label='Learning Rate')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def generate_classification_report(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str],
    save_path: str = None
) -> Dict[str, Any]:
    """Generate detailed classification report."""
    
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Classification report saved to {save_path}")
    
    return report


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)):
    """Print model architecture summary."""
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print("="*60)


def calculate_flops(model: torch.nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)):
    """Calculate FLOPs (Floating Point Operations) for the model."""
    try:
        from thop import profile
        
        input_tensor = torch.randn(input_size)
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        return flops, params
    except ImportError:
        logger.warning("thop library not available. Cannot calculate FLOPs.")
        return None, None


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Apply mixup augmentation to batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation to batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    
    # Generate random bounding box
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    image_paths: List[str],
    class_names: List[str],
    save_path: str
):
    """Save model predictions to CSV file."""
    import pandas as pd
    
    data = {
        'image_path': image_paths,
        'true_class': [class_names[t] for t in targets],
        'predicted_class': [class_names[p] for p in predictions],
        'correct': predictions == targets
    }
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Predictions saved to {save_path}")


def create_directory_structure(base_path: Path):
    """Create necessary directory structure for the project."""
    directories = [
        'data/raw',
        'data/processed/train',
        'data/processed/val',
        'data/processed/test',
        'models',
        'logs',
        'results/plots',
        'results/reports'
    ]
    
    for dir_path in directories:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Directory structure created at {base_path}")


def validate_dataset_structure(data_path: Path) -> bool:
    """Validate that dataset has proper structure."""
    required_dirs = ['train', 'val', 'test']
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            logger.error(f"Required directory not found: {dir_path}")
            return False
        
        # Check if directory has class subdirectories with images
        class_dirs = [d for d in dir_path.iterdir() if d.is_dir()]
        if not class_dirs:
            logger.error(f"No class directories found in {dir_path}")
            return False
        
        # Check for images in class directories
        has_images = False
        for class_dir in class_dirs:
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            if image_files:
                has_images = True
                break
        
        if not has_images:
            logger.error(f"No images found in {dir_path}")
            return False
    
    logger.info(f"Dataset structure validation passed for {data_path}")
    return True
