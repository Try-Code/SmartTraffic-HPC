# 🚀 Quick Start Guide

## Smart Traffic Congestion Detection System

This guide will help you get the system up and running quickly.

---

## ⚡ Quick Installation (Windows)

### 1. Install Prerequisites

```powershell
# Install Python 3.8+
winget install Python.Python.3.11

# Install CMake
winget install Kitware.CMake

# Install Visual Studio 2022 (with C++ development tools)
# Download from: https://visualstudio.microsoft.com/
```

### 2. Install OpenCV

**Option A: Using vcpkg (Recommended)**
```powershell
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install opencv[contrib]:x64-windows
.\vcpkg integrate install
```

**Option B: Pre-built binaries**
- Download from: https://opencv.org/releases/
- Extract to `C:\opencv`
- Add to PATH: `C:\opencv\build\x64\vc16\bin`

### 3. Install MS-MPI
- Download from: https://www.microsoft.com/en-us/download/details.aspx?id=105289
- Install both `msmpisetup.exe` and `msmpisdk.msi`

### 4. Clone and Setup Project

```powershell
cd "d:\New folder\Smart_traffic_CongestionUsingHPC\Smart_traffic_CongestionUsingHPC"

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download models and setup directories
python scripts\download_datasets.py
```

### 5. Build C++ Components

```powershell
# Create build directory
mkdir build
cd build

# Configure (if using vcpkg)
cmake .. -DCMAKE_TOOLCHAIN_FILE="path\to\vcpkg\scripts\buildsystems\vcpkg.cmake"

# Or configure (if OpenCV is in PATH)
cmake ..

# Build
cmake --build . --config Release

# Test build
.\Release\main_traffic_monitor.exe --help
```

---

## 🎯 Running Your First Detection

### Option 1: Python (Easiest)

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Process a single image
python src\python\realtime_processor.py --image data\images\test\sample.jpg

# Process a video
python src\python\realtime_processor.py --video data\videos\traffic.mp4

# Use webcam
python src\python\realtime_processor.py --camera 0
```

### Option 2: C++ (Faster)

```powershell
cd build\Release

# Process video with full monitoring
.\main_traffic_monitor.exe ..\..\data\videos\traffic.mp4 output.avi

# Use webcam
.\main_traffic_monitor.exe 0

# Process single image
.\vehicle_detector.exe ..\..\data\images\test\sample.jpg
```

### Option 3: Parallel Processing

**OpenMP (Multi-threaded):**
```powershell
cd build\Release

# Process batch of images with 8 threads
.\openmp_detector.exe ..\..\data\images\test ..\..\output\results 8
```

**MPI (Distributed):**
```powershell
# Process on 4 cores
mpiexec -n 4 .\mpi_batch_processor.exe ..\..\data\images\test ..\..\output\results.csv
```

---

## 📊 Sample Outputs

After running the system, you'll see:

1. **Real-time visualization** with:
   - Bounding boxes around detected vehicles
   - Vehicle tracking with IDs
   - Speed information
   - Congestion level banner
   - Live metrics panel

2. **Console output**:
   ```
   === Frame 30 ===
   Vehicles: 15
   Avg Speed: 45.2 km/h
   Congestion: 1 (Light)
   Recommendation: Light traffic. Monitor for changes.
   ```

3. **Output files**:
   - Processed video: `output.avi`
   - Statistics: `output/results/processing_results.csv`
   - Screenshots: `screenshot_*.jpg`

---

## 🎮 Keyboard Controls

When running the main traffic monitor:

- **`q`** or **`ESC`**: Quit application
- **`p`**: Pause/Resume processing
- **`s`**: Save screenshot of current frame

---

## 🔧 Common Issues & Solutions

### Issue: "OpenCV not found"

**Solution:**
```powershell
# Set OpenCV_DIR environment variable
$env:OpenCV_DIR = "C:\opencv\build"

# Or specify in CMake
cmake .. -DOpenCV_DIR="C:\opencv\build"
```

### Issue: "CUDA out of memory"

**Solution:**
```python
# In Python scripts, reduce batch size
trainer.train(batch=8)  # Instead of 16 or 32

# Or reduce image size
preprocessor = ImagePreprocessor(target_size=(416, 416))
```

### Issue: "No vehicles detected"

**Solution:**
1. Check if model is loaded correctly
2. Lower confidence threshold:
   ```cpp
   detector.detect(frame, 0.3);  // Lower from 0.5 to 0.3
   ```
3. Verify input video/image quality

### Issue: "Slow processing"

**Solutions:**
1. Use smaller YOLO model: `yolov8n.pt` instead of `yolov8x.pt`
2. Reduce input resolution
3. Enable GPU acceleration
4. Use C++ instead of Python
5. Use parallel processing (OpenMP/MPI)

---

## 📈 Performance Tips

### For Real-time Processing:
- Use YOLOv8n (nano) model
- Resize input to 640x640
- Enable GPU if available
- Use C++ implementation

### For Accuracy:
- Use YOLOv8l or YOLOv8x model
- Use higher resolution (1280x1280)
- Fine-tune on your specific dataset
- Adjust confidence thresholds

### For Batch Processing:
- Use MPI for distributed processing
- Use OpenMP for multi-threaded processing
- Process multiple videos in parallel

---

## 🎓 Next Steps

1. **Collect Your Data**
   - Record traffic videos from your cameras
   - Or download public datasets (UA-DETRAC, BDD100K)

2. **Annotate Data** (if training custom model)
   ```powershell
   # Use LabelImg or CVAT for annotation
   pip install labelImg
   labelImg
   ```

3. **Train Custom Model**
   ```powershell
   python src\python\train_yolo.py --data custom_data.yaml --epochs 100
   ```

4. **Deploy**
   - Use the trained model with main_traffic_monitor
   - Set up continuous monitoring
   - Integrate with traffic management systems

---

## 📞 Getting Help

1. Check `IMPLEMENTATION_GUIDE.md` for detailed documentation
2. Review `PROJECT_SUMMARY.md` for architecture overview
3. Look at example scripts in `scripts/`
4. Check code comments in source files

---

## 🎉 Success Checklist

- [ ] All dependencies installed
- [ ] Project builds without errors
- [ ] Can process test image
- [ ] Can process test video
- [ ] Real-time visualization works
- [ ] Output files are generated
- [ ] Performance is acceptable

If all items are checked, you're ready to start monitoring traffic! 🚗🚙🚕

---

**Happy Traffic Monitoring!** 🎯
