"""
Model evaluation and testing utilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

from config import BEST_MODEL_PATH, BASE_DIR
from models import load_pretrained_model
from data_loader import create_data_loaders
from utils import AverageMeter, accuracy, plot_confusion_matrix, generate_classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or str(BEST_MODEL_PATH)
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model = None
        self.class_mapping = None
        self.class_names = []
        
        self.results = {
            'test_accuracy': 0.0,
            'test_loss': 0.0,
            'per_class_metrics': {},
            'confusion_matrix': None,
            'predictions': [],
            'ground_truth': []
        }
    
    def load_model(self):
        """Load the trained model."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load class mapping
        class_mapping_path = BASE_DIR / "models" / "class_mapping.json"
        if not class_mapping_path.exists():
            raise FileNotFoundError(f"Class mapping not found: {class_mapping_path}")
        
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)
        
        self.class_names = list(self.class_mapping.keys())
        num_classes = len(self.class_names)
        
        # Load model
        self.model = load_pretrained_model(
            model_path=self.model_path,
            num_classes=num_classes
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded with {num_classes} classes")
    
    def evaluate_on_test_set(self):
        """Evaluate model on test dataset."""
        logger.info("Evaluating on test dataset...")
        
        # Load data
        _, _, test_loader = create_data_loaders(batch_size=32)
        
        criterion = nn.CrossEntropyLoss()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                
                # Store predictions
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Update meters
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                if len(self.class_names) >= 5:
                    top5.update(acc5[0].item(), images.size(0))
                
                pbar.set_postfix({
                    'Loss': f'{losses.avg:.4f}',
                    'Acc@1': f'{top1.avg:.2f}%',
                    'Acc@5': f'{top5.avg:.2f}%' if len(self.class_names) >= 5 else 'N/A'
                })
        
        self.results.update({
            'test_accuracy': top1.avg,
            'test_loss': losses.avg,
            'top5_accuracy': top5.avg if len(self.class_names) >= 5 else None,
            'predictions': all_predictions,
            'ground_truth': all_targets,
            'probabilities': all_probabilities
        })
        
        logger.info(f"Test Results - Accuracy: {top1.avg:.2f}%, Loss: {losses.avg:.4f}")
        
        return self.results
    
    def calculate_detailed_metrics(self):
        """Calculate detailed per-class metrics."""
        if not self.results['predictions']:
            logger.warning("No predictions available. Run evaluation first.")
            return
        
        y_true = np.array(self.results['ground_truth'])
        y_pred = np.array(self.results['predictions'])
        
        # Overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class results
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1_score': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        self.results.update({
            'overall_accuracy': overall_accuracy,
            'per_class_metrics': per_class_metrics,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            },
            'confusion_matrix': cm.tolist()
        })
        
        logger.info(f"Detailed metrics calculated. Overall accuracy: {overall_accuracy:.4f}")
    
    def save_results(self, save_dir: str = None):
        """Save evaluation results to files."""
        if not save_dir:
            save_dir = BASE_DIR / "results"
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save numerical results
        results_file = save_dir / "evaluation_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_results[key] = [item.tolist() for item in value]
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        
        # Generate and save plots
        if self.results['confusion_matrix'] is not None:
            cm_plot_path = save_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                y_true=np.array(self.results['ground_truth']),
                y_pred=np.array(self.results['predictions']),
                class_names=self.class_names,
                save_path=str(cm_plot_path),
                normalize=True
            )
        
        # Generate classification report
        if self.results['predictions']:
            report_path = save_dir / "classification_report.json"
            generate_classification_report(
                y_true=np.array(self.results['ground_truth']),
                y_pred=np.array(self.results['predictions']),
                class_names=self.class_names,
                save_path=str(report_path)
            )
    
    def print_summary(self):
        """Print evaluation summary."""
        if not self.results.get('overall_accuracy'):
            logger.warning("No evaluation results available.")
            return
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        print(f"Test Accuracy: {self.results['test_accuracy']:.2f}%")
        print(f"Test Loss: {self.results['test_loss']:.4f}")
        
        if self.results.get('top5_accuracy'):
            print(f"Top-5 Accuracy: {self.results['top5_accuracy']:.2f}%")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {self.results['overall_accuracy']:.4f}")
        
        if 'macro_avg' in self.results:
            macro = self.results['macro_avg']
            print(f"  Macro Precision: {macro['precision']:.4f}")
            print(f"  Macro Recall: {macro['recall']:.4f}")
            print(f"  Macro F1-Score: {macro['f1_score']:.4f}")
        
        if 'weighted_avg' in self.results:
            weighted = self.results['weighted_avg']
            print(f"  Weighted Precision: {weighted['precision']:.4f}")
            print(f"  Weighted Recall: {weighted['recall']:.4f}")
            print(f"  Weighted F1-Score: {weighted['f1_score']:.4f}")
        
        print(f"\nTotal Classes: {len(self.class_names)}")
        print(f"Total Test Samples: {len(self.results['ground_truth'])}")
        print("="*60)
    
    def analyze_misclassifications(self, top_k: int = 10):
        """Analyze most common misclassifications."""
        if not self.results['predictions']:
            logger.warning("No predictions available for analysis.")
            return
        
        y_true = np.array(self.results['ground_truth'])
        y_pred = np.array(self.results['predictions'])
        
        # Find misclassified samples
        misclassified_mask = y_true != y_pred
        misclassified_true = y_true[misclassified_mask]
        misclassified_pred = y_pred[misclassified_mask]
        
        # Count misclassification patterns
        misclass_patterns = {}
        for true_idx, pred_idx in zip(misclassified_true, misclassified_pred):
            true_class = self.class_names[true_idx]
            pred_class = self.class_names[pred_idx]
            pattern = f"{true_class} -> {pred_class}"
            
            misclass_patterns[pattern] = misclass_patterns.get(pattern, 0) + 1
        
        # Sort by frequency
        sorted_patterns = sorted(misclass_patterns.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_k} Misclassification Patterns:")
        print("-" * 50)
        for i, (pattern, count) in enumerate(sorted_patterns[:top_k]):
            print(f"{i+1:2d}. {pattern}: {count} times")
        
        return sorted_patterns
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("Starting complete model evaluation...")
        
        # Load model
        self.load_model()
        
        # Run evaluation
        self.evaluate_on_test_set()
        
        # Calculate detailed metrics
        self.calculate_detailed_metrics()
        
        # Print summary
        self.print_summary()
        
        # Analyze misclassifications
        self.analyze_misclassifications()
        
        # Save results
        self.save_results()
        
        logger.info("Evaluation completed successfully!")
        
        return self.results


def compare_models(model_paths: List[str], model_names: List[str] = None):
    """Compare multiple trained models."""
    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(len(model_paths))]
    
    results_comparison = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        logger.info(f"Evaluating {model_name}...")
        
        evaluator = ModelEvaluator(model_path=model_path)
        results = evaluator.run_complete_evaluation()
        
        results_comparison[model_name] = {
            'test_accuracy': results['test_accuracy'],
            'overall_accuracy': results.get('overall_accuracy', 0),
            'macro_f1': results.get('macro_avg', {}).get('f1_score', 0),
            'weighted_f1': results.get('weighted_avg', {}).get('f1_score', 0)
        }
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Test Acc':<12} {'Overall Acc':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-"*80)
    
    for name, metrics in results_comparison.items():
        print(f"{name:<20} {metrics['test_accuracy']:<12.2f} "
              f"{metrics['overall_accuracy']:<12.4f} {metrics['macro_f1']:<12.4f} "
              f"{metrics['weighted_f1']:<12.4f}")
    
    print("="*80)
    
    return results_comparison


def main():
    """Main evaluation function."""
    evaluator = ModelEvaluator()
    results = evaluator.run_complete_evaluation()
    
    return results


if __name__ == "__main__":
    main()
