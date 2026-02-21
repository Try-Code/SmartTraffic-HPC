# 🚀 GETTING STARTED - Quick Setup Guide

## Smart Traffic Congestion Detection System

This guide will get you up and running in **30 minutes**!

---

## ⚡ Quick Setup (3 Steps)

### Step 1: Install Python Dependencies (5 minutes)

```bash
# Navigate to project directory
cd "d:/New folder/Smart_traffic_CongestionUsingHPC/Smart_traffic_CongestionUsingHPC"

# Install Python packages
pip install -r requirements.txt
```

### Step 2: Download Models & Setup Structure (10 minutes)

```bash
# Run setup script
python scripts/download_datasets.py
```

This will:
- ✅ Create directory structure
- ✅ Download YOLOv8 models
- ✅ Provide dataset links

### Step 3: Test the System (15 minutes)

```bash
# Test with a sample image (if you have one)
python src/python/realtime_processor.py

# Or try the complete pipeline
python scripts/run_pipeline.py --mode inference --source data/images/test
```

---

## 📝 What You Have Now

### ✅ Complete File Structure
```
✓ 25+ source files
✓ C++ implementations (OpenCV, MPI, OpenMP, CUDA)
✓ Python scripts (YOLOv8, preprocessing, real-time)
✓ Configuration files
✓ Documentation (README, guides, examples)
```

### ✅ Key Components

1. **Image Processing** (`src/image_processing/`)
   - `vehicle_detector.cpp` - Detect vehicles using YOLO/Caffe
   - `density_estimator.cpp` - Estimate traffic density
   - Complete with headers in `include/`

2. **Python Modules** (`src/python/`)
   - `train_yolo.py` - Train YOLOv8 models
   - `image_preprocessor.py` - Preprocess images
   - `realtime_processor.py` - Real-time video processing

3. **HPC Implementations** (`src/parallel/`)
   - `openmp_detector.cpp` - Multi-threaded processing
   - `mpi_image_batch.cpp` - Distributed processing
   - `cuda_preprocessing.cu` - GPU acceleration

4. **Utilities** (`src/utils/`)
   - `image_utils.cpp` - Image processing utilities

---

## 🎯 Choose Your Path

### Path A: Python Developer (Easiest)
**Best if you want to:** Quickly test and prototype

```bash
# 1. Process a video
python src/python/realtime_processor.py

# 2. Train a model
python src/python/train_yolo.py

# 3. Preprocess images
python src/python/image_preprocessor.py
```

**Requirements:** Python 3.8+, pip packages

---

### Path B: C++ Developer (Performance)
**Best if you want to:** Maximum performance

```bash
# 1. Build the project
mkdir build && cd build
cmake ..
cmake --build . --config Release

# 2. Run vehicle detector
./vehicle_detector ../data/images/test/sample.jpg

# 3. Run parallel processing
./openmp_detector ../data/images/test ../output/results 8
```

**Requirements:** CMake, C++ compiler, OpenCV

---

### Path C: HPC Researcher (Scalability)
**Best if you want to:** Process large datasets

```bash
# 1. Build MPI components
cd build
cmake --build . --target mpi_batch_processor

# 2. Run distributed processing
mpirun -np 4 ./mpi_batch_processor ../data/images/test ../output/results.csv
```

**Requirements:** MPI, cluster access

---

### Path D: Complete Pipeline (Recommended)
**Best if you want to:** End-to-end solution

```bash
# Run the complete pipeline
python scripts/run_pipeline.py --mode full
```

This will:
1. ✅ Preprocess images
2. ✅ Run inference
3. ✅ Generate visualizations
4. ✅ Save statistics

---

## 📊 What to Do Next

### Option 1: Use Pre-trained Models (Fastest)
```bash
# Models are auto-downloaded to models/
# Just run inference on your images/videos

python src/python/realtime_processor.py --source YOUR_VIDEO.mp4
```

### Option 2: Train on Custom Data
```bash
# 1. Prepare your dataset
# - Add images to data/images/train/
# - Add annotations to data/annotations/

# 2. Train
python src/python/train_yolo.py

# 3. Use your trained model
# Model saved to: output/yolo_training/traffic_detector/weights/best.pt
```

### Option 3: Process Existing Datasets
```bash
# Download public datasets:
# - UA-DETRAC: https://detrac-db.rit.albany.edu/
# - BDD100K: https://www.bdd100k.com/
# - KITTI: http://www.cvlibs.net/datasets/kitti/

# Then process them:
python scripts/run_pipeline.py --mode inference --source path/to/dataset
```

---

## 🎓 Learning Resources

### Understand the Code
1. **Start with:** `README.md` - Project overview
2. **Then read:** `IMPLEMENTATION_GUIDE.md` - Detailed guide
3. **Reference:** `PROJECT_SUMMARY.md` - Quick reference

### Example Workflows

**Workflow 1: Process a Single Image**
```python
from src.python.realtime_processor import RealtimeTrafficProcessor

processor = RealtimeTrafficProcessor()
result, metrics = processor.process_image('image.jpg', 'output.jpg')
print(f"Vehicles: {metrics['vehicle_count']}")
```

**Workflow 2: Process a Video**
```python
processor = RealtimeTrafficProcessor()
processor.process_video('traffic.mp4', 'output.mp4', display=True)
```

**Workflow 3: Batch Process Images**
```bash
# Using OpenMP (C++)
./build/openmp_detector data/images/test output/results 8
```

---

## 🐛 Common Issues & Solutions

### Issue: "No module named 'ultralytics'"
```bash
# Solution:
pip install ultralytics
```

### Issue: "OpenCV not found"
```bash
# Windows:
pip install opencv-python opencv-contrib-python

# Linux:
sudo apt install python3-opencv
```

### Issue: "CUDA out of memory"
```python
# Solution: Reduce batch size
trainer.train(batch=8)  # Instead of 16 or 32
```

### Issue: "No images found"
```bash
# Solution: Add images to the correct directory
# Place images in: data/images/test/
# Or specify custom path in commands
```

---

## 📈 Performance Tips

### For Faster Processing:
1. **Use smaller model**: `yolov8n.pt` instead of `yolov8x.pt`
2. **Reduce resolution**: 640x640 instead of 1280x1280
3. **Use GPU**: Set `device='cuda'` if available
4. **Parallel processing**: Use OpenMP or MPI for batch jobs

### For Better Accuracy:
1. **Use larger model**: `yolov8l.pt` or `yolov8x.pt`
2. **Higher resolution**: 1280x1280
3. **Train on custom data**: Fine-tune on your specific use case
4. **More training epochs**: 100+ epochs

---

## 🎯 Success Checklist

After setup, you should be able to:

- [ ] Run Python scripts without errors
- [ ] Process a sample image
- [ ] See vehicle detections
- [ ] View congestion levels
- [ ] Save results to output/

**If you can do all of the above, you're ready to go! 🎉**

---

## 📞 Need Help?

1. **Check Documentation:**
   - `README.md` - Main documentation
   - `IMPLEMENTATION_GUIDE.md` - Detailed guide
   - `PROJECT_SUMMARY.md` - Quick reference

2. **Review Examples:**
   - Look at code in `src/python/`
   - Check configuration in `config/`

3. **Common Commands:**
   ```bash
   # List all Python scripts
   dir src\\python\\*.py
   
   # Check installed packages
   pip list
   
   # Test Python import
   python -c "import cv2; import torch; print('OK')"
   ```

---

## 🚀 Ready to Start!

You now have a **complete, production-ready** traffic congestion detection system!

**Recommended First Steps:**
1. ✅ Download a sample traffic video from YouTube or Pexels
2. ✅ Place it in `data/videos/`
3. ✅ Run: `python src/python/realtime_processor.py`
4. ✅ Watch the magic happen! 🎬

**Happy Traffic Monitoring! 🚗🚙🚕**

---

*For detailed implementation, see IMPLEMENTATION_GUIDE.md*
*For project overview, see README.md*
*For quick reference, see PROJECT_SUMMARY.md*
