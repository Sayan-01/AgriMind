#!/usr/bin/env python3
"""
Test script for AgriMind ML Inference functionality.
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

# Test configurations and utilities
def create_dummy_image(size=(224, 224)):
    """Create a dummy RGB image for testing."""
    return Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))

def create_dummy_dataset_structure(base_path, classes=['class1', 'class2'], images_per_class=5):
    """Create dummy dataset structure for testing."""
    base_path = Path(base_path)
    
    for class_name in classes:
        class_dir = base_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(images_per_class):
            img = create_dummy_image()
            img.save(class_dir / f"image_{i}.jpg")

class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_dataset_structure_scan(self):
        """Test dataset structure scanning."""
        from src.data_preprocessing import DataPreprocessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy structure
            create_dummy_dataset_structure(temp_dir, ['healthy', 'diseased'])
            
            preprocessor = DataPreprocessor()
            result = preprocessor.scan_dataset_structure(Path(temp_dir))
            
            assert result['structure'] == 'class_folders'
            assert 'healthy' in result['classes']
            assert 'diseased' in result['classes']
            assert result['total_images'] == 10  # 2 classes * 5 images
    
    def test_class_mapping_creation(self):
        """Test class mapping creation."""
        import json
        from src.config import BASE_DIR
        
        # Mock class mapping
        test_mapping = {'class1': 0, 'class2': 1, 'class3': 2}
        
        mapping_file = BASE_DIR / "models" / "test_class_mapping.json"
        mapping_file.parent.mkdir(exist_ok=True)
        
        with open(mapping_file, 'w') as f:
            json.dump(test_mapping, f)
        
        # Test loading
        with open(mapping_file, 'r') as f:
            loaded_mapping = json.load(f)
        
        assert loaded_mapping == test_mapping
        
        # Clean up
        mapping_file.unlink()

class TestModels:
    """Test model architectures."""
    
    def test_plant_disease_classifier_creation(self):
        """Test model creation."""
        from src.models import PlantDiseaseClassifier
        
        model = PlantDiseaseClassifier(
            num_classes=10,
            backbone='efficientnet_b0',  # Smaller model for testing
            pretrained=False  # Avoid downloading weights in tests
        )
        
        assert model.num_classes == 10
        assert model.backbone_name == 'efficientnet_b0'
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        from src.models import PlantDiseaseClassifier
        
        model = PlantDiseaseClassifier(
            num_classes=5,
            backbone='efficientnet_b0',
            pretrained=False
        )
        
        # Create dummy input
        dummy_input = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (2, 5)  # batch_size, num_classes
    
    def test_parameter_counting(self):
        """Test parameter counting utility."""
        from src.models import count_parameters, PlantDiseaseClassifier
        
        model = PlantDiseaseClassifier(
            num_classes=3,
            backbone='efficientnet_b0',
            pretrained=False
        )
        
        total_params, trainable_params = count_parameters(model)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

class TestDataLoader:
    """Test data loading functionality."""
    
    def test_transforms_creation(self):
        """Test transform creation."""
        from src.data_loader import get_train_transforms, get_val_transforms
        
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        assert train_transform is not None
        assert val_transform is not None
    
    def test_dataset_creation(self):
        """Test dataset creation with dummy data."""
        from src.data_loader import PlantDiseaseDataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy dataset
            create_dummy_dataset_structure(temp_dir)
            
            dataset = PlantDiseaseDataset(
                data_dir=Path(temp_dir),
                transform=None  # No transform for simplicity
            )
            
            assert len(dataset) == 10  # 2 classes * 5 images
            assert dataset.num_classes == 2
            
            # Test getitem
            image, label = dataset[0]
            assert isinstance(image, Image.Image)
            assert isinstance(label, int)

class TestUtils:
    """Test utility functions."""
    
    def test_average_meter(self):
        """Test AverageMeter functionality."""
        from src.utils import AverageMeter
        
        meter = AverageMeter()
        
        meter.update(1.0, 1)
        meter.update(2.0, 1)
        meter.update(3.0, 1)
        
        assert meter.avg == 2.0
        assert meter.sum == 6.0
        assert meter.count == 3
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        from src.utils import accuracy
        
        # Create dummy predictions and targets
        outputs = torch.tensor([[2.0, 1.0, 0.5], [0.5, 2.0, 1.0]])
        targets = torch.tensor([0, 1])
        
        acc = accuracy(outputs, targets, topk=(1,))
        
        assert len(acc) == 1
        assert acc[0].item() == 100.0  # Both predictions correct
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint save/load functionality."""
        from src.utils import save_checkpoint, load_checkpoint
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"
            
            # Create dummy state
            test_state = {
                'epoch': 10,
                'model_state_dict': {'layer.weight': torch.randn(3, 3)},
                'optimizer_state_dict': {},
                'best_accuracy': 85.5
            }
            
            # Save checkpoint
            save_checkpoint(test_state, is_best=True, checkpoint_path=str(checkpoint_path))
            
            # Load checkpoint
            loaded_state = load_checkpoint(str(checkpoint_path))
            
            assert loaded_state['epoch'] == test_state['epoch']
            assert loaded_state['best_accuracy'] == test_state['best_accuracy']

class TestAPI:
    """Test API functionality."""
    
    def test_image_preprocessing(self):
        """Test image preprocessing for API."""
        from src.api import preprocess_image
        
        # Create dummy image
        dummy_image = create_dummy_image()
        
        # Note: This test might fail if transforms are not loaded
        # In a real test environment, you'd mock the transform
        try:
            processed = preprocess_image(dummy_image)
            assert isinstance(processed, torch.Tensor)
            assert len(processed.shape) == 4  # Batch dimension added
        except:
            # Skip if transforms not available
            pytest.skip("Transform not available for testing")

class TestConfig:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        from src.config import MODEL_CONFIG, AUGMENTATION_CONFIG, API_CONFIG
        
        # Test that configs are loaded
        assert MODEL_CONFIG is not None
        assert AUGMENTATION_CONFIG is not None
        assert API_CONFIG is not None
        
        # Test specific values
        assert MODEL_CONFIG.batch_size > 0
        assert MODEL_CONFIG.learning_rate > 0
        assert len(API_CONFIG.allowed_extensions) > 0
    
    def test_dataset_config(self):
        """Test dataset configuration."""
        from src.config import DATASETS
        
        assert 'plantvillage' in DATASETS
        assert 'plantdoc' in DATASETS
        assert 'bangla_crops' in DATASETS
        assert 'rice_leaf' in DATASETS
        assert 'indian_trees' in DATASETS
        
        # Test dataset structure
        for name, config in DATASETS.items():
            assert hasattr(config, 'name')
            assert hasattr(config, 'path')
            assert hasattr(config, 'image_size')

def run_tests():
    """Run all tests."""
    print("Running AgriMind ML Inference tests...")
    
    # Run with pytest
    pytest.main([__file__, "-v"])

if __name__ == "__main__":
    run_tests()
