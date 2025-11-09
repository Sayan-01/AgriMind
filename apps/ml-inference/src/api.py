"""
FastAPI application for plant disease detection inference.
"""

import io
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import API_CONFIG, BEST_MODEL_PATH, CLASS_MAPPING_PATH
from models import load_pretrained_model
from data_loader import get_val_transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AgriMind Plant Disease Detection API",
    description="AI-powered plant disease detection and diagnosis system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessing
model = None
class_mapping = None
transform = None
device = None

def load_model_and_preprocessing():
    """Load the trained model and preprocessing pipeline."""
    global model, class_mapping, transform, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load class mapping
        if not CLASS_MAPPING_PATH.exists():
            raise FileNotFoundError(f"Class mapping not found: {CLASS_MAPPING_PATH}")
        
        with open(CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        
        num_classes = len(class_mapping)
        logger.info(f"Loaded class mapping with {num_classes} classes")
        
        # Load model
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {BEST_MODEL_PATH}")
        
        model = load_pretrained_model(
            model_path=str(BEST_MODEL_PATH),
            num_classes=num_classes
        )
        model = model.to(device)
        model.eval()
        
        # Load preprocessing transforms
        transform = get_val_transforms()
        
        logger.info("Model and preprocessing loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        if transform:
            if hasattr(transform, 'apply'):
                # Albumentations transform
                image_np = np.array(image)
                transformed = transform(image=image_np)
                image_tensor = transformed['image']
            else:
                # PyTorch transforms
                image_tensor = transform(image)
        else:
            # Fallback preprocessing
            image = image.resize((224, 224))
            image_np = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        
        # Add batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(device)
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise


def postprocess_predictions(outputs: torch.Tensor, top_k: int = 5) -> Dict[str, Any]:
    """Convert model outputs to human-readable predictions."""
    try:
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_mapping)))
        
        # Convert to lists
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Create reverse mapping (index to class name)
        idx_to_class = {idx: class_name for class_name, idx in class_mapping.items()}
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = idx_to_class.get(idx, f"Unknown_{idx}")
            predictions.append({
                "class": class_name,
                "confidence": float(prob),
                "percentage": float(prob * 100)
            })
        
        return {
            "predictions": predictions,
            "top_prediction": predictions[0],
            "total_classes": len(class_mapping)
        }
        
    except Exception as e:
        logger.error(f"Prediction postprocessing failed: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    try:
        load_model_and_preprocessing()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {e}")
        # Don't raise here to allow the API to start even without a model
        # This is useful for development/testing


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AgriMind Plant Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": str(device) if device else "unknown",
        "classes_loaded": len(class_mapping) if class_mapping else 0
    }


@app.get("/info")
async def get_info():
    """Get API and model information."""
    if not model or not class_mapping:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        },
        "classes": list(class_mapping.keys()),
        "num_classes": len(class_mapping),
        "device": str(device),
        "config": {
            "max_image_size": API_CONFIG.max_image_size,
            "allowed_extensions": API_CONFIG.allowed_extensions
        }
    }


@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Predict plant disease from uploaded image.
    
    Args:
        file: Image file (jpg, png, etc.)
        top_k: Number of top predictions to return (default: 5)
    
    Returns:
        JSON response with predictions and confidence scores
    """
    
    # Validate model availability
    if not model or not class_mapping:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in API_CONFIG.allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {API_CONFIG.allowed_extensions}"
        )
    
    try:
        # Read and validate image
        contents = await file.read()
        
        if len(contents) > API_CONFIG.max_image_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {API_CONFIG.max_image_size / 1024 / 1024:.1f}MB"
            )
        
        # Load image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Preprocess image
        input_tensor = preprocess_image(image)
        
        # Model inference
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Postprocess results
        result = postprocess_predictions(outputs, top_k)
        
        # Add metadata
        result.update({
            "filename": file.filename,
            "image_size": image.size,
            "processing_info": {
                "device": str(device),
                "input_shape": list(input_tensor.shape)
            }
        })
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict plant diseases for multiple images.
    
    Args:
        files: List of image files
    
    Returns:
        JSON response with predictions for each image
    """
    
    if not model or not class_mapping:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file similar to single prediction
            contents = await file.read()
            
            if len(contents) > API_CONFIG.max_image_size:
                results.append({
                    "filename": file.filename,
                    "error": "File too large",
                    "success": False
                })
                continue
            
            image = Image.open(io.BytesIO(contents))
            input_tensor = preprocess_image(image)
            
            with torch.no_grad():
                outputs = model(input_tensor)
            
            result = postprocess_predictions(outputs, top_k=3)  # Limit to top 3 for batch
            result.update({
                "filename": file.filename,
                "success": True,
                "batch_index": i
            })
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False,
                "batch_index": i
            })
    
    return JSONResponse(content={
        "batch_results": results,
        "total_files": len(files),
        "successful_predictions": sum(1 for r in results if r.get("success", False))
    })


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=API_CONFIG.host,
        port=API_CONFIG.port,
        workers=API_CONFIG.workers,
        reload=True
    )
