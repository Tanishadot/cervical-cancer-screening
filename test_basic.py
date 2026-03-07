"""
Basic test script to verify pipeline structure without heavy dependencies.
"""

import os
import sys

def test_project_structure():
    """Test that all required directories and files exist."""
    print("Testing project structure...")
    
    required_dirs = [
        'datasets',
        'models', 
        'preprocessing',
        'training',
        'xai',
        'utils',
        'configs',
        'outputs'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'setup.py',
        'configs/config.yaml',
        'utils/label_mapping.py',
        'utils/config_manager.py',
        'utils/visualization.py',
        'datasets/dataset.py',
        'preprocessing/transforms.py',
        'preprocessing/image_preprocessing.py',
        'preprocessing/stain_normalization.py',
        'models/model_factory.py',
        'training/trainer.py',
        'training/cross_dataset_eval.py',
        'xai/grad_cam.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required directories and files exist")
    return True

def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\nTesting Python syntax...")
    
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        return False
    
    print(f"✅ All {len(python_files)} Python files have valid syntax")
    return True

def test_imports_basic():
    """Test basic imports without heavy dependencies."""
    print("\nTesting basic imports...")
    
    try:
        # Test utility imports
        sys.path.append('.')
        from utils.label_mapping import LabelMapper, ClassificationMode
        from utils.config_manager import ConfigManager
        print("✅ Utility imports successful")
        
        # Test basic functionality
        mapper = LabelMapper(ClassificationMode.BINARY)
        assert mapper.get_num_classes() == 2
        print("✅ Label mapping functionality works")
        
        # Test config manager
        config_manager = ConfigManager()
        assert config_manager.get_experiment_config() is not None
        print("✅ Configuration manager works")
        
        return True
    except Exception as e:
        print(f"❌ Basic import test failed: {e}")
        return False

def test_config_file():
    """Test configuration file validity."""
    print("\nTesting configuration file...")
    
    try:
        import yaml
        
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['experiment', 'dataset', 'model', 'training']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        print("✅ Configuration file is valid")
        return True
    except Exception as e:
        print(f"❌ Configuration file test failed: {e}")
        return False

def test_requirements():
    """Test requirements file format."""
    print("\nTesting requirements file...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.readlines()
        
        # Check that requirements are properly formatted
        for req in requirements:
            req = req.strip()
            if req and not req.startswith('#'):
                if '==' not in req and '>=' not in req and '<=' not in req and '>' not in req and '<' not in req:
                    print(f"⚠️  Warning: Requirement may need version specification: {req}")
        
        print(f"✅ Requirements file has {len([r for r in requirements if r.strip() and not r.startswith('#')])} packages")
        return True
    except Exception as e:
        print(f"❌ Requirements file test failed: {e}")
        return False

def run_basic_tests():
    """Run all basic tests."""
    print("="*60)
    print("RUNNING BASIC PIPELINE TESTS")
    print("="*60)
    
    tests = [
        test_project_structure,
        test_python_syntax,
        test_imports_basic,
        test_config_file,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"BASIC TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All basic tests passed! Pipeline structure is correct.")
        print("\nNext steps:")
        print("1. Install PyTorch and other dependencies: pip install -r requirements.txt")
        print("2. Prepare your datasets in the datasets/ directory")
        print("3. Run the pipeline: python main.py")
        return True
    else:
        print("⚠️  Some basic tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)
