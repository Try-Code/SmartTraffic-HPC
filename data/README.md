# Traffic Dataset

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
