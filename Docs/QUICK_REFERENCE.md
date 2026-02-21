# ⚡ Quick Reference Card

## 🚀 One-Minute Start

```powershell
# 1. Navigate to project
cd "C:\Users\prashant jee\OneDrive\Desktop\Shefa\Smart_traffic_CongestionUsingHPC"

# 2. Activate environment
.\.venv\Scripts\Activate.ps1

# 3. Run real-time detection
python src/python/realtime_processor.py --camera 0
```

## 📦 Install Everything

```powershell
cd "C:\Users\prashant jee\OneDrive\Desktop\Shefa\Smart_traffic_CongestionUsingHPC"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_datasets.py
```

## 🔨 Build C++ (Advanced)

```powershell
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 📤 Push to GitHub

```powershell
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/Smart-Traffic-Congestion-HPC.git
git branch -M main
git push -u origin main
```

## 📝 Key Files

| File | Purpose |
|------|---------|
| `src/python/realtime_processor.py` | Main processing script |
| `config/traffic_data.yaml` | YOLO configuration |
| `requirements.txt` | Python dependencies |
| `CMakeLists.txt` | C++ build configuration |
| `README.md` | Project documentation |
| `.gitignore` | Git exclusions |

## 🎮 Usage Examples

```powershell
# Webcam
python src/python/realtime_processor.py --camera 0

# Image file
python src/python/realtime_processor.py --image "image.jpg"

# Video file
python src/python/realtime_processor.py --video "traffic.mp4"

# Train model
python src/python/train_yolo.py --data config/traffic_data.yaml
```

## ✅ Verify Setup

```powershell
python verify_setup.py
```

## 📊 Project Structure

```
├── src/
│   ├── python/              # Python scripts
│   ├── image_processing/    # C++ image modules
│   └── parallel/            # HPC implementations
├── models/                  # Pre-trained models
├── data/                    # Datasets
├── config/                  # Configuration files
└── output/                  # Results
```

## 🌐 GitHub URLs (After Push)

```
Repository: https://github.com/USERNAME/Smart-Traffic-Congestion-HPC
Documentation: https://username.github.io/Smart-Traffic-Congestion-HPC/
```

## 🆘 Help

- **Setup Help**: See [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **GitHub Help**: See [GITHUB_PUSH_GUIDE.md](GITHUB_PUSH_GUIDE.md)
- **Project Info**: See [README.md](README.md)
- **Full Deployment**: See [DEPLOYMENT_COMPLETE.md](DEPLOYMENT_COMPLETE.md)

---

**Remember:** Replace `USERNAME` with your GitHub username! 🚀
