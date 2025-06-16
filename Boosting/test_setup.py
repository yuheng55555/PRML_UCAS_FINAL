#!/usr/bin/env python3
"""
Project Setup Test Script
Used to verify that all dependencies are correctly installed and the project can run normally
"""

import sys
import importlib
from typing import List, Tuple

def test_package_imports() -> Tuple[bool, List[str]]:
    """Test if all necessary packages can be imported normally"""
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'torch',
        'torchvision',
        'xgboost',
        'lightgbm',
        'catboost',
        'cv2',
        'skimage',
        'PIL'
    ]
    
    failed_imports = []
    
    print("Testing dependency package imports...")
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package:<15} - Import successful")
        except ImportError as e:
            failed_imports.append(package)
            print(f"✗ {package:<15} - Import failed: {str(e)}")
    
    if failed_imports:
        print(f"\n❌ {len(failed_imports)} packages failed to import:")
        for package in failed_imports:
            print(f"  - {package}")
        print("\nPlease run: pip install -r requirements.txt")
        return False, failed_imports
    else:
        print(f"\n✅ All {len(required_packages)} dependency packages imported successfully!")
        return True, []

def test_project_modules() -> Tuple[bool, List[str]]:
    """Test if project modules can be imported normally"""
    project_modules = [
        'data_preprocessing',
        'models', 
        'train_evaluate',
        'visualization'
    ]
    
    failed_modules = []
    
    print("Testing project module imports...")
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module:<20} - Import successful")
        except ImportError as e:
            failed_modules.append(module)
            print(f"✗ {module:<20} - Import failed: {str(e)}")
    
    if failed_modules:
        print(f"\n❌ {len(failed_modules)} project modules failed to import:")
        for module in failed_modules:
            print(f"  - {module}")
        return False, failed_modules
    else:
        print(f"\n✅ All {len(project_modules)} project modules imported successfully!")
        return True, []

def test_basic_functionality() -> bool:
    """Test basic functionality"""
    try:
        print("Testing basic functionality...")
        
        # Test data loader
        from data_preprocessing import CIFAR10DataLoader
        print("✓ CIFAR10DataLoader class can be imported normally")
        
        # Test model classes
        from models import XGBoostModel, LightGBMModel, CatBoostModel
        print("✓ Boosting model classes can be imported normally")
        
        # Test training pipeline
        from train_evaluate import TrainingPipeline
        print("✓ TrainingPipeline class can be imported normally")
        
        # Test visualization
        from visualization import ModelVisualizer
        print("✓ ModelVisualizer class can be imported normally")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("CIFAR-10 Boosting Project - Setup Test")
    print("=" * 50)
    
    # Test Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Python version too low, requires Python 3.7+")
        return False
    else:
        print("✅ Python version meets requirements")
    
    # Run all tests
    all_tests_passed = True
    
    # Test package imports
    packages_ok, failed_packages = test_package_imports()
    all_tests_passed = all_tests_passed and packages_ok
    
    # Test project modules
    modules_ok, failed_modules = test_project_modules()
    all_tests_passed = all_tests_passed and modules_ok
    
    # Test basic functionality
    if packages_ok and modules_ok:
        functionality_ok = test_basic_functionality()
        all_tests_passed = all_tests_passed and functionality_ok
    
    # Final result
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! Project setup is correct and ready to use.")
        print("\nRun the project:")
        print("  Quick test: python main.py --quick_test")
        print("  Full training: python main.py")
    else:
        print("❌ Some tests failed, please check the error messages above and fix them.")
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 