#!/usr/bin/env python3
"""
Diagnostic script to check model loading and weights
"""

import torch
from simple_predictor import SimplePredictor, PlantDiseaseClassifier
from simple_config import NUM_CLASSES, DEFAULT_MODEL_PATH
import numpy as np

def main():
    print("🔧 Model Diagnostic")
    print("=" * 50)
    
    # Check if model file exists and its size
    if DEFAULT_MODEL_PATH.exists():
        size_mb = DEFAULT_MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Model file exists: {size_mb:.1f} MB")
    else:
        print("❌ Model file not found!")
        return
    
    # Load raw checkpoint
    print("\n📦 Loading raw checkpoint...")
    checkpoint = torch.load(DEFAULT_MODEL_PATH, map_location='cpu')
    
    print(f"📊 Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"📋 Keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"🔢 State dict has {len(state_dict)} parameters")
    
    # Check final layer dimensions
    if 'head.fc.weight' in state_dict:
        fc_weight = state_dict['head.fc.weight']
        print(f"🎯 Final layer shape: {fc_weight.shape}")
        print(f"   Expected: ({NUM_CLASSES}, 1920)")
        
        # Check if weights are reasonable
        weight_mean = fc_weight.mean().item()
        weight_std = fc_weight.std().item()
        print(f"📈 Weight stats - Mean: {weight_mean:.6f}, Std: {weight_std:.6f}")
        
        if abs(weight_mean) < 1e-6 and weight_std < 1e-6:
            print("⚠️  Weights appear to be zeros - model might not be trained!")
    
    # Test model creation and loading
    print("\n🏗️  Testing model creation...")
    model = PlantDiseaseClassifier(num_classes=NUM_CLASSES)
    print(f"✅ Model created successfully")
    
    # Create predictor and test
    print("\n🔮 Testing predictor...")
    try:
        predictor = SimplePredictor()
        print(f"✅ Predictor created successfully")
        print(f"📊 Number of classes: {len(predictor.class_mapping)}")
        
        # Test on random input
        print("\n🎲 Testing with random input...")
        dummy_input = torch.randn(1, 3, 224, 224)
        
        predictor.model.eval()
        with torch.no_grad():
            output = predictor.model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            
        print(f"🎯 Output shape: {output.shape}")
        print(f"📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"🎲 Probabilities sum: {probabilities.sum().item():.6f}")
        
        # Show top predictions for random input
        top_probs, top_indices = torch.topk(probabilities, k=5)
        print(f"\n🔝 Top 5 predictions for random input:")
        for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            class_name = predictor.idx_to_class[idx.item()]
            print(f"   {i+1}. {class_name}: {prob.item():.1%}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n✅ Diagnostic complete")

if __name__ == "__main__":
    main()
