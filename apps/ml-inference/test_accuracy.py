#!/usr/bin/env python3
"""
Test accuracy of the plant disease model on test images
"""

import os
from pathlib import Path
from simple_predictor import SimplePredictor
from collections import defaultdict
import json

def extract_true_label(filename):
    """Extract the true disease label from filename"""
    # Remove extension and numbers
    name = Path(filename).stem
    # Extract disease name (everything before the last number sequence)
    import re
    match = re.match(r'([A-Za-z_]+)', name)
    if match:
        disease = match.group(1).rstrip('_')
        return disease
    return name.split('_')[0] if '_' in name else name

def find_closest_class(true_label, class_names):
    """Find the closest matching class name"""
    true_lower = true_label.lower()
    
    # Direct matches
    for class_name in class_names:
        if true_lower in class_name.lower() or class_name.lower() in true_lower:
            return class_name
    
    # Partial matches
    for class_name in class_names:
        if any(word in class_name.lower() for word in true_lower.split('_')):
            return class_name
    
    return None

def main():
    print("🧪 Testing Plant Disease Model Accuracy")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SimplePredictor()
    test_dir = Path("test_images")
    
    if not test_dir.exists():
        print("❌ Test images directory not found!")
        return
    
    # Get all test images
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No test images found!")
        return
    
    print(f"📸 Found {len(image_files)} test images")
    print(f"🔬 Model has {len(predictor.class_mapping)} classes")
    print("\n📋 Available classes:")
    for i, class_name in enumerate(predictor.idx_to_class.values()):
        print(f"   {i}: {class_name}")
    
    # Test each image
    results = []
    correct_predictions = 0
    total_predictions = len(image_files)
    
    print(f"\n🔍 Testing {total_predictions} images...")
    print("-" * 80)
    
    disease_stats = defaultdict(lambda: {"total": 0, "correct": 0, "predictions": []})
    
    for i, img_path in enumerate(image_files, 1):
        try:
            # Get prediction
            result = predictor.predict(img_path, top_k=3)
            
            # Extract true label from filename
            true_label = extract_true_label(img_path.name)
            predicted_label = result['predicted_class']
            confidence = result['confidence']
            
            # Find if there's a matching class
            closest_class = find_closest_class(true_label, predictor.idx_to_class.values())
            
            # Determine if prediction is correct
            is_correct = False
            if closest_class:
                is_correct = predicted_label == closest_class
                if is_correct:
                    correct_predictions += 1
            
            # Store stats
            disease_stats[true_label]["total"] += 1
            disease_stats[true_label]["predictions"].append({
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": is_correct
            })
            if is_correct:
                disease_stats[true_label]["correct"] += 1
            
            # Display result
            status = "✅" if is_correct else "❌"
            print(f"{status} [{i:2d}/{total_predictions}] {img_path.name:25} | True: {true_label:15} | Pred: {predicted_label:25} | Conf: {confidence:.1%}")
            
            results.append({
                "image": img_path.name,
                "true_label": true_label,
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": is_correct,
                "closest_class": closest_class
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    # Calculate and display statistics
    print("\n" + "=" * 80)
    print("📊 ACCURACY RESULTS")
    print("=" * 80)
    
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"🎯 Overall Accuracy: {correct_predictions}/{total_predictions} = {overall_accuracy:.1f}%")
    
    print(f"\n📈 Per-Disease Statistics:")
    print("-" * 50)
    for disease, stats in disease_stats.items():
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"   {disease:20} | {stats['correct']:2d}/{stats['total']:2d} = {accuracy:5.1f}%")
    
    # Show confidence distribution
    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    print(f"\n📊 Average Confidence: {avg_confidence:.1%}")
    
    # High confidence predictions
    high_conf = [r for r in results if r['confidence'] > 0.8]
    print(f"🎯 High Confidence (>80%): {len(high_conf)}/{len(results)} = {len(high_conf)/len(results)*100:.1f}%")
    
    # Save detailed results
    with open("test_results.json", "w") as f:
        json.dump({
            "overall_accuracy": overall_accuracy,
            "total_images": total_predictions,
            "correct_predictions": correct_predictions,
            "per_disease_stats": dict(disease_stats),
            "detailed_results": results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: test_results.json")
    
    return overall_accuracy

if __name__ == "__main__":
    accuracy = main()
    print(f"\n🏁 Test completed with {accuracy:.1f}% accuracy")
