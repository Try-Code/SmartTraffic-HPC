"""
Dataset Download Script
Downloads public traffic datasets for training and testing
"""

import os
import gdown
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import tarfile
import shutil


class DatasetDownloader:
    """Download and prepare traffic datasets"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, output_path, extract=True):
        """Download file from URL"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading from {url}")
        
        try:
            # Download
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=output_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"✅ Downloaded to {output_path}")
            
            # Extract if archive
            if extract and (output_path.suffix in ['.zip', '.tar', '.gz', '.tgz']):
                self.extract_archive(output_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return False
    
    def download_from_gdrive(self, file_id, output_path, extract=True):
        """Download from Google Drive"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading from Google Drive (ID: {file_id})")
        
        try:
            gdown.download(id=file_id, output=str(output_path), quiet=False)
            print(f"✅ Downloaded to {output_path}")
            
            if extract and (output_path.suffix in ['.zip', '.tar', '.gz']):
                self.extract_archive(output_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return False
    
    def extract_archive(self, archive_path):
        """Extract archive file"""
        archive_path = Path(archive_path)
        extract_dir = archive_path.parent / archive_path.stem
        
        print(f"📦 Extracting {archive_path.name}...")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            print(f"✅ Extracted to {extract_dir}")
            
            # Remove archive
            archive_path.unlink()
            print(f"🗑️  Removed archive file")
            
        except Exception as e:
            print(f"❌ Error extracting: {e}")
    
    def download_coco_vehicles(self):
        """Download COCO dataset (vehicle classes only)"""
        print("\n🚗 Downloading COCO Vehicle Dataset...")
        
        # COCO 2017 Train images
        coco_train_url = "http://images.cocodataset.org/zips/train2017.zip"
        coco_val_url = "http://images.cocodataset.org/zips/val2017.zip"
        coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        # Note: Full COCO dataset is very large (25GB+)
        # For demo purposes, we'll provide instructions instead
        print("⚠️  COCO dataset is very large (25GB+)")
        print("   Download manually from: https://cocodataset.org/#download")
        print("   Or use the sample dataset below")
    
    def download_sample_dataset(self):
        """Download a smaller sample dataset for testing"""
        print("\n📸 Creating sample dataset...")
        
        # Create sample structure
        sample_dirs = [
            self.data_dir / 'images' / 'train' / 'congested',
            self.data_dir / 'images' / 'train' / 'free_flow',
            self.data_dir / 'images' / 'validation',
            self.data_dir / 'images' / 'test'
        ]
        
        for dir_path in sample_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Sample directory structure created")
        print("\n📝 Next steps:")
        print("   1. Add your traffic images to the created directories")
        print("   2. Or download from these public sources:")
        print("      - UA-DETRAC: https://detrac-db.rit.albany.edu/")
        print("      - BDD100K: https://www.bdd100k.com/")
        print("      - Cityscapes: https://www.cityscapes-dataset.com/")
        print("      - KITTI: http://www.cvlibs.net/datasets/kitti/")
    
    def download_pretrained_models(self):
        """Download pre-trained YOLO models"""
        print("\n🤖 Downloading pre-trained models...")
        
        models_dir = self.data_dir.parent / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLOv8 models
        yolo_models = {
            'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        }
        
        for model_name, url in yolo_models.items():
            model_path = models_dir / model_name
            if not model_path.exists():
                print(f"\n📥 Downloading {model_name}...")
                self.download_file(url, model_path, extract=False)
            else:
                print(f"✅ {model_name} already exists")
    
    def download_sample_videos(self):
        """Download sample traffic videos"""
        print("\n🎥 Sample traffic videos...")
        print("   Download free traffic videos from:")
        print("   - Pexels: https://www.pexels.com/search/videos/traffic/")
        print("   - Pixabay: https://pixabay.com/videos/search/traffic/")
        print("   - YouTube (with proper licensing)")
        
        videos_dir = self.data_dir / 'videos'
        videos_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Place videos in: {videos_dir}")
    
    def setup_dataset_structure(self):
        """Create complete dataset structure"""
        print("\n📁 Setting up dataset structure...")
        
        directories = [
            'images/train/congested',
            'images/train/free_flow',
            'images/validation',
            'images/test',
            'processed/train',
            'processed/validation',
            'processed/test',
            'videos',
            'annotations',
        ]
        
        for dir_path in directories:
            full_path = self.data_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Dataset structure created")
        
        # Create README
        readme_path = self.data_dir / 'README.md'
        readme_content = """# Traffic Dataset

## Directory Structure

- `images/train/` - Training images
  - `congested/` - Images with traffic congestion
  - `free_flow/` - Images with free-flowing traffic
- `images/validation/` - Validation images
- `images/test/` - Test images
- `processed/` - Preprocessed images
- `videos/` - Traffic video files
- `annotations/` - YOLO format annotations

## Data Sources

### Recommended Datasets

1. **UA-DETRAC** (Vehicle Detection and Tracking)
   - URL: https://detrac-db.rit.albany.edu/
   - Size: ~40GB
   - Contains: 100 videos, 140,000 frames

2. **BDD100K** (Berkeley DeepDrive)
   - URL: https://www.bdd100k.com/
   - Size: ~100GB
   - Contains: Diverse driving scenarios

3. **KITTI** (Autonomous Driving)
   - URL: http://www.cvlibs.net/datasets/kitti/
   - Size: ~15GB
   - Contains: Urban traffic scenes

4. **Cityscapes**
   - URL: https://www.cityscapes-dataset.com/
   - Size: ~11GB
   - Contains: Urban street scenes

### Free Video Sources

- Pexels: https://www.pexels.com/search/videos/traffic/
- Pixabay: https://pixabay.com/videos/search/traffic/

## Annotation Format

Use YOLO format for annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values normalized to [0, 1]

## Classes

0: car
1: bus
2: truck
3: motorbike
4: bicycle
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"📝 Created README at {readme_path}")


def main():
    """Main download pipeline"""
    
    print("=" * 60)
    print("  Traffic Dataset Downloader")
    print("=" * 60)
    
    downloader = DatasetDownloader(data_dir='data')
    
    # Setup structure
    downloader.setup_dataset_structure()
    
    # Download pre-trained models
    downloader.download_pretrained_models()
    
    # Create sample dataset
    downloader.download_sample_dataset()
    
    # Show video sources
    downloader.download_sample_videos()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("   1. Download datasets from the sources listed above")
    print("   2. Place images in data/images/train/, validation/, test/")
    print("   3. Add annotations in YOLO format to data/annotations/")
    print("   4. Run preprocessing: python src/python/image_preprocessor.py")
    print("   5. Train model: python src/python/train_yolo.py")
    print("\n💡 Tip: Start with a small subset of data for testing!")


if __name__ == '__main__':
    main()
