# ==================================================
# SETUP GUIDE: Smart Traffic Congestion Detection
# ==================================================

## STEP 1: Install Required Software (Windows)

### A. Python 3.8+ (if not already installed)
```powershell
# Check if Python is installed
python --version

# If not, install via winget or from python.org
winget install Python.Python.3.11
```

### B. CMake 3.15+ (if not already installed)
```powershell
# Check if CMake is installed
cmake --version

# If not, install via winget
winget install Kitware.CMake
```

### C. Visual Studio 2022 Build Tools (if not already installed)
- Download from: https://visualstudio.microsoft.com/downloads/
- Install "Desktop development with C++"

### D. OpenCV (Choose ONE option)

#### Option 1: Install via vcpkg (RECOMMENDED)
```powershell
# Clone vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap vcpkg
.\bootstrap-vcpkg.bat

# Install OpenCV with contrib modules
.\vcpkg install opencv[contrib]:x64-windows

# Integrate with CMake
.\vcpkg integrate install
```

#### Option 2: Pre-built Binaries
- Download from: https://opencv.org/releases/
- Extract to `C:\opencv`
- Add `C:\opencv\build\x64\vc16\bin` to PATH

### E. MS-MPI (Optional, for distributed processing)
- Download from: https://www.microsoft.com/en-us/download/details.aspx?id=105289
- Install both `msmpisetup.exe` and `msmpisdk.msi`

---

## STEP 2: Setup Project Environment (Automated)

```powershell
# Navigate to project directory
cd "C:\Users\prashant jee\OneDrive\Desktop\Shefa\Smart_traffic_CongestionUsingHPC"

# The virtual environment should already be created
# Verify with:
dir .venv

# Activate it (if needed)
.\.venv\Scripts\activate

# Python dependencies are already installed
# Verify with:
pip list | findstr torch opencv ultralytics

# Download models and setup directories
python scripts\download_datasets.py
```

---

## STEP 3: Build C++ Components

```powershell
# From project root directory
$projectDir = "C:\Users\prashant jee\OneDrive\Desktop\Shefa\Smart_traffic_CongestionUsingHPC"
cd $projectDir

# Create build directory
mkdir build
cd build

# If using vcpkg (RECOMMENDED)
$vcpkgPath = "C:\path\to\vcpkg"  # Update this path
cmake .. -DCMAKE_TOOLCHAIN_FILE="$vcpkgPath\scripts\buildsystems\vcpkg.cmake" -DCMAKE_BUILD_TYPE=Release

# OR if OpenCV is in system PATH
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Test the build
.\Release\vehicle_detector.exe --help
```

---

## STEP 4: Run Python Examples

```powershell
# Activate environment
.\.venv\Scripts\activate

# Test real-time processor (requires a video or image)
python src\python\realtime_processor.py --image "data\images\test\sample.jpg"

# Or use webcam (camera ID 0)
python src\python\realtime_processor.py --camera 0

# Or process a video
python src\python\realtime_processor.py --video "data\videos\traffic.mp4"
```

---

## STEP 5: Run C++ Executables

```powershell
cd build\Release

# Vehicle detection
.\vehicle_detector.exe "C:\path\to\image.jpg"

# Density estimation
.\density_estimator.exe "C:\path\to\image.jpg"

# Congestion analysis
.\congestion_analyzer.exe "C:\path\to\image.jpg"

# MPI batch processing (requires MS-MPI)
mpiexec -n 4 .\mpi_batch.exe "C:\path\to\images\directory"
```

---

## TROUBLESHOOTING

### Issue: CMake can't find OpenCV
**Solution:** 
- Use vcpkg method (recommended)
- Or set `OpenCV_DIR` environment variable before running cmake
- Or use `cmake .. -DOpenCV_DIR="C:\path\to\opencv\build"`

### Issue: CUDA not found
**Solution:**
- CUDA is optional. If not needed, edit CMakeLists.txt and remove `CUDA` from `find_package()`
- Or install CUDA Toolkit 11.0+

### Issue: MPI compilation errors
**Solution:**
- MS-MPI is optional. Install it or modify CMakeLists.txt
- Or skip MPI components

### Issue: Python dependencies installation fails
**Solution:**
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Try again
pip install -r requirements.txt

# If torch fails, use CPU version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## NEXT STEPS: Push to GitHub

Once the project runs locally:

1. **Initialize Git locally:**
   ```powershell
   cd "C:\Users\prashant jee\OneDrive\Desktop\Shefa\Smart_traffic_CongestionUsingHPC"
   git init
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   ```

2. **Create GitHub repository:**
   - Go to https://github.com/new
   - Create repository named: `Smart-Traffic-Congestion-HPC`
   - Do NOT initialize with README (we have one)

3. **Push code:**
   ```powershell
   git add .
   git commit -m "Initial commit: Smart Traffic Congestion Detection System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/Smart-Traffic-Congestion-HPC.git
   git push -u origin main
   ```

4. **Enable GitHub Pages:**
   - Go to repository Settings → Pages
   - Set source to "main" branch, "/root" folder
   - Documentation will be available at: https://your-username.github.io/Smart-Traffic-Congestion-HPC/

---

**Status:** Setup guide complete. Follow steps 1-5 to get started!
