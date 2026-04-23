"""
Test script to verify the cervical cancer classification pipeline functionality.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.label_mapping import LabelMapper, get_label_mapper
        from utils.config_manager import ConfigManager, load_config
        from utils.visualization import Visualizer
        from datasets.dataset import create_dataset, get_dataset_info
        from preprocessing.transforms import get_train_transforms, get_val_transforms
        from preprocessing.image_preprocessing import ImagePreprocessor
        from preprocessing.stain_normalization import MacenkoNormalizer
        from models.model_factory import create_model, ModelFactory
        from training.trainer import Trainer, MetricsCalculator
        from training.cross_dataset_eval import CrossDatasetEvaluator
        from xai.grad_cam import create_xai_analyzer
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_label_mapping():
    """Test label mapping functionality."""
    print("\nTesting label mapping...")
    
    try:
        from utils.label_mapping import LabelMapper, ClassificationMode
        
        # Test binary classification
        binary_mapper = LabelMapper(ClassificationMode.BINARY)
        assert binary_mapper.get_num_classes() == 2
        assert binary_mapper.get_class_names()[0] == "NORMAL"
        assert binary_mapper.get_class_names()[1] == "ABNORMAL"
        
        # Test multiclass mapping
        multiclass_mapper = LabelMapper(ClassificationMode.MULTICLASS)
        from utils.label_mapping import DatasetType
        multiclass_classes = multiclass_mapper.get_class_names(DatasetType.SIPAKMED)
        assert len(multiclass_classes) == 5
        
        print("✅ Label mapping tests passed")
        return True
    except Exception as e:
        print(f"❌ Label mapping test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        device = torch.device('cpu')  # Use CPU for testing
        
        # Test EfficientNet-B0
        model1 = create_model('efficientnet_b0', num_classes=2, pretrained=False)
        assert model1 is not None
        
        # Test ResNet50
        model2 = create_model('resnet50', num_classes=5, pretrained=False)
        assert model2 is not None
        
        # Test Swin Transformer
        model3 = create_model('swin_transformer', num_classes=2, pretrained=False)
        assert model3 is not None
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output1 = model1(dummy_input)
            output2 = model2(dummy_input)
            output3 = model3(dummy_input)
        
        assert output1.shape[1] == 2
        assert output2.shape[1] == 5
        assert output3.shape[1] == 2
        
        print("✅ Model creation tests passed")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_transforms():
    """Test data augmentation transforms."""
    print("\nTesting transforms...")
    
    try:
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        # Test training transforms
        train_transform = get_train_transforms()
        transformed = train_transform(image=np.array(pil_image))
        assert 'image' in transformed
        assert transformed['image'].shape == (3, 224, 224)
        
        # Test validation transforms
        val_transform = get_val_transforms()
        transformed = val_transform(image=np.array(pil_image))
        assert 'image' in transformed
        assert transformed['image'].shape == (3, 224, 224)
        
        print("✅ Transform tests passed")
        return True
    except Exception as e:
        print(f"❌ Transform test failed: {e}")
        return False

def test_preprocessing():
    """Test image preprocessing."""
    print("\nTesting preprocessing...")
    
    try:
        from preprocessing.image_preprocessing import ImagePreprocessor
        from preprocessing.stain_normalization import MacenkoNormalizer
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Test image preprocessor
        preprocessor = ImagePreprocessor(target_size=(224, 224))
        processed = preprocessor.preprocess(dummy_image)
        assert processed.shape == (224, 224, 3)
        assert processed.dtype == np.uint8
        
        # Test stain normalizer
        target_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        normalizer = MacenkoNormalizer()
        normalizer.fit(target_image)
        normalized = normalizer.normalize(dummy_image)
        assert normalized.shape == dummy_image.shape
        assert normalized.dtype == np.uint8
        
        print("✅ Preprocessing tests passed")
        return True
    except Exception as e:
        print(f"❌ Preprocessing test failed: {e}")
        return False

def test_config_manager():
    """Test configuration manager."""
    print("\nTesting configuration manager...")
    
    try:
        # Test loading default config
        config_manager = load_config()
        assert config_manager is not None
        assert config_manager.get_experiment_config() is not None
        
        # Test config updates
        updates = {
            'experiment': {'name': 'test_experiment'},
            'model': {'architecture': 'resnet50'}
        }
        config_manager.update_config(updates)
        
        config = config_manager.get_experiment_config()
        assert config.name == 'test_experiment'
        assert config.model_architecture == 'resnet50'
        
        print("✅ Configuration manager tests passed")
        return True
    except Exception as e:
        print(f"❌ Configuration manager test failed: {e}")
        return False

def test_metrics_calculator():
    """Test metrics calculation."""
    print("\nTesting metrics calculator...")
    
    try:
        from training.trainer import MetricsCalculator
        
        # Create dummy data
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.9, 0.1], [0.6, 0.4]])
        
        # Test metrics calculation
        calculator = MetricsCalculator(num_classes=2, class_names=['Class_0', 'Class_1'])
        metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'auc_roc' in metrics
        
        print("✅ Metrics calculator tests passed")
        return True
    except Exception as e:
        print(f"❌ Metrics calculator test failed: {e}")
        return False

def test_visualization():
    """Test visualization utilities."""
    print("\nTesting visualization...")
    
    try:
        from utils.visualization import Visualizer
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Test visualizer creation
        visualizer = Visualizer(output_dir=temp_dir)
        
        # Test plotting functions with dummy data
        history = {
            'train_loss': [0.8, 0.6, 0.4],
            'val_loss': [0.9, 0.7, 0.5],
            'train_accuracy': [0.7, 0.8, 0.85],
            'val_accuracy': [0.65, 0.75, 0.8]
        }
        
        fig = visualizer.plot_training_curves(history, save_fig=False)
        assert fig is not None
        
        # Test confusion matrix
        cm = np.array([[50, 10], [5, 35]])
        class_names = ['Normal', 'Abnormal']
        fig = visualizer.plot_confusion_matrix(cm, class_names, save_fig=False)
        assert fig is not None
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        print("✅ Visualization tests passed")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_xai_analyzer():
    """Test XAI analyzer."""
    print("\nTesting XAI analyzer...")
    
    try:
        from xai.grad_cam import create_xai_analyzer
        
        # Create simple model for testing
        model = create_model('efficientnet_b0', num_classes=2, pretrained=False)
        device = torch.device('cpu')
        
        # Create XAI analyzer
        analyzer = create_xai_analyzer(model, device, output_dir="test_xai")
        assert analyzer is not None
        
        print("✅ XAI analyzer tests passed")
        return True
    except Exception as e:
        print(f"❌ XAI analyzer test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RUNNING CERVICAL CANCER PIPELINE TESTS")
    print("="*60)
    
    tests = [
        test_imports,
        test_label_mapping,
        test_model_creation,
        test_transforms,
        test_preprocessing,
        test_config_manager,
        test_metrics_calculator,
        test_visualization,
        test_xai_analyzer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
