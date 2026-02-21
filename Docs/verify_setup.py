#!/usr/bin/env python
"""
Test script to verify the Smart Traffic project setup
Run with: python verify_setup.py
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def check_import(module_name, display_name=None):
    """Check if a Python module is installed"""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {display_name} is installed")
        return True
    except ImportError:
        print(f"✗ {display_name} is NOT installed")
        return False

def main():
    print_header("Smart Traffic Congestion Detection - Setup Verification")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ ERROR: Python 3.8+ is required!")
        return False
    else:
        print("✓ Python version is compatible\n")
    
    # Check critical packages
    print_header("Checking Core Dependencies")
    
    all_ok = True
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("torchvision", "TorchVision")
    all_ok &= check_import("cv2", "OpenCV")
    all_ok &= check_import("ultralytics", "Ultralytics (YOLOv8)")
    all_ok &= check_import("numpy", "NumPy")
    all_ok &= check_import("pandas", "Pandas")
    all_ok &= check_import("PIL", "Pillow")
    
    # Check optional packages
    print_header("Checking Optional Dependencies")
    
    all_ok &= check_import("sklearn", "scikit-learn")
    all_ok &= check_import("matplotlib", "Matplotlib")
    all_ok &= check_import("seaborn", "Seaborn")
    
    # Check data directories
    print_header("Checking Project Structure")
    
    project_root = Path(__file__).parent
    required_dirs = [
        "data/images",
        "data/videos",
        "data/processed",
        "models",
        "output",
        "src/python",
        "src/image_processing",
        "include",
        "config"
    ]
    
    dirs_ok = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/ exists")
        else:
            print(f"✗ {dir_path}/ NOT found (will be created on first run)")
            dirs_ok = False
    
    # Check Python scripts
    print_header("Checking Python Scripts")
    
    required_scripts = [
        "src/python/realtime_processor.py",
        "src/python/train_yolo.py",
        "src/python/image_preprocessor.py",
        "scripts/download_datasets.py",
    ]
    
    scripts_ok = True
    for script_path in required_scripts:
        full_path = project_root / script_path
        if full_path.exists():
            print(f"✓ {script_path} exists")
        else:
            print(f"✗ {script_path} NOT found")
            scripts_ok = False
    
    # Check config files
    print_header("Checking Configuration Files")
    
    config_files = [
        "requirements.txt",
        "CMakeLists.txt",
        "README.md",
        "config/traffic_data.yaml",
        "config/processing_config.json"
    ]
    
    config_ok = True
    for config_file in config_files:
        full_path = project_root / config_file
        if full_path.exists():
            print(f"✓ {config_file} exists")
        else:
            print(f"✗ {config_file} NOT found")
            config_ok = False
    
    # Test imports and basic functionality
    print_header("Testing Basic Functionality")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        all_ok = False
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        all_ok = False
    
    try:
        from ultralytics import YOLO
        print("✓ YOLOv8 import successful")
        print("  (Note: Models will be downloaded on first use)")
    except Exception as e:
        print(f"✗ YOLOv8 test failed: {e}")
        all_ok = False
    
    # Final status
    print_header("Setup Summary")
    
    if all_ok and config_ok and scripts_ok:
        print("✓ ALL CHECKS PASSED!")
        print("\nYour environment is ready to run the project!")
        print("\nNext steps:")
        print("1. Run: python scripts/download_datasets.py")
        print("2. Run: python src/python/realtime_processor.py --camera 0")
        print("\nOr build C++ components:")
        print("1. mkdir build && cd build")
        print("2. cmake .. -DCMAKE_BUILD_TYPE=Release")
        print("3. cmake --build . --config Release")
        return True
    else:
        print("✗ Some checks failed. See details above.")
        print("\nTo fix issues:")
        print("1. Reinstall dependencies: pip install -r requirements.txt --upgrade")
        print("2. Check that Python 3.8+ is being used")
        print("3. If on Windows, ensure build tools are installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
