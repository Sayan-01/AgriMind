"""
Simple Plant Disease Inference Script
Simplified version for easy testing and deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
import json
import argparse
from pathlib import Path
from simple_config import DEFAULT_MODEL_PATH, CLASS_MAPPING_PATH, MODEL_NAME, NUM_CLASSES, INPUT_SIZE

class PlantDiseaseClassifier(nn.Module):
    """Simple plant disease classifier using timm backbone"""
    
    def __init__(self, num_classes=NUM_CLASSES, backbone=MODEL_NAME, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class SimplePredictor:
    """Simplified predictor class"""
    
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path or DEFAULT_MODEL_PATH
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Load model
        self.model = self._load_model()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_class_mapping(self):
        """Load class mapping from JSON file"""
        if CLASS_MAPPING_PATH.exists():
            with open(CLASS_MAPPING_PATH, 'r') as f:
                mapping = json.load(f)
                # Filter to get only the first 17 classes if more exist
                if len(mapping) > NUM_CLASSES:
                    # Sort by value and take first NUM_CLASSES
                    sorted_items = sorted(mapping.items(), key=lambda x: x[1])
                    filtered_mapping = {k: v for k, v in sorted_items[:NUM_CLASSES]}
                    return filtered_mapping
                return mapping
        else:
            # Create a default mapping for 17 classes based on common plant diseases
            default_classes = [
                "Rice_Brown_Spot", "Rice_Healthy", "Rice_Leaf_Blast", 
                "Corn_Common_Rust", "Corn_Healthy", "Corn_Northern_Leaf_Blight",
                "Potato_Early_Blight", "Potato_Healthy", "Potato_Late_Blight",
                "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
                "Tomato_Healthy", "Wheat_Brown_Rust", "Wheat_Healthy", 
                "Pepper_Bacterial_spot", "Pepper_Healthy"
            ]
            class_mapping = {cls: i for i, cls in enumerate(default_classes)}
            # Save the mapping
            CLASS_MAPPING_PATH.parent.mkdir(exist_ok=True)
            with open(CLASS_MAPPING_PATH, 'w') as f:
                json.dump(class_mapping, f, indent=2)
            return class_mapping
    
    def _load_model(self):
        """Load the trained model"""
        model = PlantDiseaseClassifier(num_classes=len(self.class_mapping))
        
        if self.model_path.exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Fix key mismatch - the checkpoint doesn't have backbone prefix
            # but our model wraps timm model in backbone
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    # Remove backbone prefix if it exists
                    new_key = key.replace('backbone.', '')
                    fixed_state_dict[f"backbone.{new_key}"] = value
                else:
                    # Add backbone prefix if it doesn't exist
                    fixed_state_dict[f"backbone.{key}"] = value
            
            model.load_state_dict(fixed_state_dict, strict=False)
        else:
            print(f"Warning: Model file {self.model_path} not found. Using random weights.")
        
        model.eval()
        model.to(self.device)
        return model
    
    def predict(self, image_path, top_k=3):
        """Predict disease from image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.class_mapping)))
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.idx_to_class[idx.item()]
            predictions.append({
                'class': class_name,
                'confidence': prob.item()
            })
        
        return {
            'predicted_class': predictions[0]['class'],
            'confidence': predictions[0]['confidence'],
            'top_predictions': predictions
        }

def main():
    parser = argparse.ArgumentParser(description='Simple Plant Disease Predictor')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--model', help='Path to model file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = SimplePredictor(args.model)
    
    # Make prediction
    try:
        result = predictor.predict(args.image_path, args.top_k)
        
        print(f"\n🔍 Analysis Results for: {Path(args.image_path).name}")
        print("=" * 50)
        print(f"🎯 Predicted Disease: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        
        if len(result['top_predictions']) > 1:
            print(f"\n📈 Top {len(result['top_predictions'])} Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['class']}: {pred['confidence']:.1%}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
