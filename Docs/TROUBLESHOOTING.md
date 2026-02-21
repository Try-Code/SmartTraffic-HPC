# 🆘 Troubleshooting & FAQ

## Common Issues & Solutions

### ❌ "Python not found"

**Problem**: `python: The term 'python' is not recognized`

**Solutions**:
```powershell
# 1. Use full path to virtual environment Python
"C:\Users\prashant jee\OneDrive\Desktop\Shefa\Smart_traffic_CongestionUsingHPC\.venv\Scripts\python.exe" --version

# 2. Or activate virtual environment first
.\.venv\Scripts\Activate.ps1
python --version

# 3. If virtual environment doesn't work, use installed Python
python --version

# 4. If python not in PATH, install from https://www.python.org/downloads/
```

---

### ❌ "ModuleNotFoundError: No module named 'torch'"

**Problem**: PyTorch or other packages not installed

**Solutions**:
```powershell
# 1. Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# 2. Check if packages are installed
pip list | findstr torch

# 3. Reinstall requirements
pip install -r requirements.txt --upgrade

# 4. If torch installation fails on Windows, try:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### ❌ "CMake not found"

**Problem**: `cmake: The term 'cmake' is not recognized`

**Solutions**:
```powershell
# 1. Check if installed
cmake --version

# 2. Install CMake
winget install Kitware.CMake

# 3. Or download from https://cmake.org/download/

# 4. Add to PATH if not automatic (Windows):
# Go to Settings → Environment Variables
# Add: C:\Program Files\CMake\bin
```

---

### ❌ "Cannot find OpenCV"

**Problem**: CMake build fails with "Could NOT find OpenCV"

**Solutions**:
```powershell
# Method 1: Use vcpkg (RECOMMENDED)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install opencv[contrib]:x64-windows
.\vcpkg integrate install

# Then build with:
cd path\to\project\build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake"

# Method 2: Manual
set OpenCV_DIR=C:\opencv\build
cmake ..

# Method 3: Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"
```

---

### ❌ "git command not recognized"

**Problem**: `git: The term 'git' is not recognized`

**Solutions**:
```powershell
# 1. Check if installed
git --version

# 2. Install Git
winget install Git.Git

# 3. Or download from https://git-scm.com/download/win

# 4. Restart PowerShell after installation
```

---

### ❌ Virtual Environment Issues

**Problem**: `.venv\Scripts\Activate.ps1` doesn't work

**Solutions**:
```powershell
# 1. Check execution policy
Get-ExecutionPolicy

# 2. If restricted, set to RemoteSigned
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Try alternative activation
.\.venv\Scripts\Activate.bat     # For Command Prompt
source .venv/bin/activate        # For WSL/Git Bash

# 4. Recreate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 5. Verify virtual environment
python -c "import sys; print(sys.prefix)"
```

---

### ❌ "Permission denied" when pushing to GitHub

**Problem**: `fatal: could not read Username`

**Solutions**:
```powershell
# 1. Use Personal Access Token (recommended)
# Generate at: https://github.com/settings/tokens
# When prompted for password, use the token

# 2. Or setup SSH keys
ssh-keygen -t ed25519 -C "your.email@gmail.com"
# Then add public key to https://github.com/settings/ssh

# 3. Store credentials
git config --global credential.helper store

# 4. Configure email
git config --global user.email "your.email@gmail.com"
git config --global user.name "Your Name"
```

---

### ⚠️ CUDA/GPU Issues

**Problem**: "CUDA out of memory" or GPU not detected

**Solutions**:
```python
# 1. Check if GPU is available
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# 2. If not available, use CPU
# Set environment variable before running script:
# set CUDA_VISIBLE_DEVICES=-1

# 3. Or modify code:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 4. Use smaller models
# In realtime_processor.py, use 'yolov8n' instead of 'yolov8m'

# 5. Use batch size 1
# Modify config/processing_config.json: "batch_size": 1
```

---

### ⚠️ Video/Camera Issues

**Problem**: "Failed to open camera" or video file not found

**Solutions**:
```powershell
# 1. Check camera is available
python src/python/realtime_processor.py --camera 0

# 2. Try different camera IDs
python src/python/realtime_processor.py --camera 1

# 3. Check video file path (use absolute path)
python src/python/realtime_processor.py --video "C:\path\to\video.mp4"

# 4. Check if FFmpeg is installed
pip install imageio-ffmpeg

# 5. Try different video formats
# Supported: .mp4, .avi, .mov, .mkv
```

---

### ⚠️ Slow Performance

**Problem**: Processing is very slow

**Solutions**:
```powershell
# 1. Use smaller YOLOv8 model (nano instead of medium)
# In code: YOLO('yolov8n.pt')

# 2. Use GPU if available
import torch
print(torch.cuda.is_available())

# 3. Enable FP16 (faster inference)
# In processing script: model = YOLO('yolov8n.pt').half()

# 4. Reduce frame size/resolution
# In config: "frame_width": 640, "frame_height": 480

# 5. Process every Nth frame
# In code: if frame_count % 3 == 0: process()

# 6. Use C++ implementation (faster)
# Build with CMake and use vehicle_detector.exe
```

---

## FAQ - Frequently Asked Questions

### Q: Do I need a GPU?
**A**: No, the project works on CPU. GPU (CUDA) is optional for faster processing.

### Q: Can I run this on macOS/Linux?
**A**: Yes! Same instructions apply. Use `source .venv/bin/activate` instead of `.\.venv\Scripts\Activate.ps1`

### Q: What's the minimum Python version?
**A**: Python 3.8+. Python 3.10+ recommended.

### Q: Can I use different YOLOv8 models?
**A**: Yes! Available models:
- `yolov8n` - nano (fastest, smallest)
- `yolov8s` - small
- `yolov8m` - medium
- `yolov8l` - large (most accurate, slowest)

### Q: How do I train a custom model?
**A**: Use `src/python/train_yolo.py` with your labeled dataset in YOLO format.

### Q: How much disk space do I need?
**A**: 
- Base project: ~100 MB
- Models: ~300 MB
- Data/Output: Depends on dataset (typically 1-10 GB)

### Q: Is the project production-ready?
**A**: It's a research/demo project. For production, consider:
- Error handling
- Logging
- Input validation
- Security (especially for web deployment)
- Performance optimization

### Q: How do I contribute to the project?
**A**: 
1. Fork the repository on GitHub
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and commit
4. Push to your fork and create a Pull Request

### Q: Where can I find more help?
**A**: 
- YOLOv8 docs: https://docs.ultralytics.com/
- PyTorch docs: https://pytorch.org/docs/
- OpenCV docs: https://docs.opencv.org/
- GitHub support: https://support.github.com/

---

## Performance Benchmarks

Typical performance on different hardware:

| Hardware | Model | Speed | Memory |
|----------|-------|-------|--------|
| CPU (Intel i7) | YOLOv8n | ~30 FPS | 500 MB |
| CPU (Intel i7) | YOLOv8m | ~5 FPS | 1.5 GB |
| GPU (NVIDIA RTX 3060) | YOLOv8n | ~150 FPS | 2 GB |
| GPU (NVIDIA RTX 3060) | YOLOv8m | ~80 FPS | 4 GB |

*Note: Actual performance depends on image size, batch size, and system load*

---

## Debug Mode

To enable detailed logging:

```powershell
# Set environment variables
$env:OPENCV_LOG_LEVEL=DEBUG
$env:ULTRALYTICS_DEBUG=True

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run
python src/python/realtime_processor.py --camera 0
```

---

**Can't find your issue?** Check:
1. [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup
2. [README.md](README.md) for project overview
3. [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md) for full deployment guide

**Still stuck?** Open an issue on GitHub with:
- Error message
- Steps to reproduce
- Your system info (OS, Python version, GPU if applicable)
