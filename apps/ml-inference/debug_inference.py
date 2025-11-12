#!/usr/bin/env python3
"""
Debug inference to see raw logits and understand model behavior
"""

import torch
from simple_predictor import SimplePredictor
from PIL import Image
import torch.nn.functional as F

def debug_inference(image_path):
    """Debug inference with raw logits"""
    predictor = SimplePredictor()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)
    
    print(f"🔍 Debugging: {image_path}")
    print(f"📊 Input tensor shape: {input_tensor.shape}")
    print(f"📈 Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Get raw model output (logits)
    with torch.no_grad():
        logits = predictor.model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
    
    print(f"\n🎯 Raw logits:")
    print(f"   Range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"   Mean: {logits.mean():.3f}")
    print(f"   Std: {logits.std():.3f}")
    
    print(f"\n📊 All class probabilities:")
    for i in range(len(predictor.class_mapping)):
        class_name = predictor.idx_to_class[i]
        prob = probabilities[0, i].item()
        logit = logits[0, i].item()
        print(f"   {i:2d}. {class_name:30} | Logit: {logit:8.3f} | Prob: {prob:6.1%}")
    
    # Check if there's extreme softmax saturation
    max_prob = probabilities.max().item()
    if max_prob > 0.999:
        print(f"\n⚠️  Extreme softmax saturation detected! Max prob: {max_prob:.6f}")
        print("    This suggests very large logit differences.")

def main():
    test_images = [
        "test_images/Brown_rust1000.jpg",
        "test_images/Brown_rust148.jpg",
    ]
    
    for img_path in test_images:
        debug_inference(img_path)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
