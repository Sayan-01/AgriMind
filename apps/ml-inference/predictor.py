"""
Updated Plant Disease Predictor
Drop-in replacement using Hugging Face model
"""

from huggingface_predictor import HuggingFacePredictor
import argparse
from pathlib import Path

def predict_disease(image_path: str, top_k: int = 3):
    """
    Main prediction function - drop-in replacement for the old predictor
    
    Args:
        image_path: Path to the image file
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with prediction results
    """
    # Initialize the HuggingFace predictor
    predictor = HuggingFacePredictor()
    
    # Get prediction
    result = predictor.infer(image_path)
    
    # Format result to match the old predictor interface
    formatted_result = {
        'predicted_class': result['label'],
        'confidence': result['confidence'],
        'top_predictions': [{
            'class': result['label'],
            'confidence': result['confidence']
        }],
        'model_type': 'HuggingFace_ViT',
        'image_path': image_path
    }
    
    return formatted_result

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Plant Disease Predictor (HuggingFace)')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    
    args = parser.parse_args()
    
    try:
        result = predict_disease(args.image_path, args.top_k)
        
        print(f"\n🔍 Analysis Results for: {Path(args.image_path).name}")
        print("=" * 50)
        print(f"🎯 Predicted Disease: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"🤖 Model: {result['model_type']}")
        
        # Additional disease information
        disease = result['predicted_class']
        print(f"\n📋 Disease Information:")
        
        if 'Brown_Rust' in disease:
            print("   🌾 Brown Rust (Leaf Rust)")
            print("   🔸 Crop: Wheat")
            print("   🔸 Severity: Moderate to High")
            print("   🔸 Treatment: Fungicide application recommended")
            print("   🔸 Prevention: Resistant varieties, proper spacing")
        elif 'Healthy' in disease:
            print("   ✅ Healthy Plant")
            print("   🔸 No disease detected")
            print("   🔸 Continue regular monitoring")
        elif 'Blight' in disease:
            print("   🍂 Blight Disease")
            print("   🔸 Severity: High")
            print("   🔸 Treatment: Remove affected leaves, fungicide")
        elif 'Spot' in disease:
            print("   🔴 Leaf Spot Disease")
            print("   🔸 Severity: Moderate")
            print("   🔸 Treatment: Improve air circulation, fungicide")
        else:
            print("   🔍 Check agricultural resources for specific treatment")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
