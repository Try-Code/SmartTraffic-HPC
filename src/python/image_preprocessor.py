"""
Image Preprocessing Module for Traffic Analysis
Handles image augmentation, normalization, and preparation
"""

import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm


class ImagePreprocessor:
    """Image preprocessing for traffic images"""
    
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        self.augmentation_pipeline = self.create_augmentation_pipeline()
        
    def create_augmentation_pipeline(self, is_training=True):
        """Create augmentation pipeline using Albumentations"""
        
        if is_training:
            return A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.3),
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.2),
                A.RandomFog(p=0.1),
                A.RandomRain(p=0.1),
                A.RandomSunFlare(p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            return A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def preprocess_image(self, image_path, augment=False):
        """
        Preprocess a single image
        
        Args:
            image_path: Path to image
            augment: Apply augmentation
            
        Returns:
            Preprocessed image
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation
        if augment:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image']
        else:
            # Simple resize and normalize
            image = cv2.resize(image, self.target_size)
            image = image.astype(np.float32) / 255.0
            return image
    
    def batch_preprocess(self, input_dir, output_dir, augment=False):
        """
        Preprocess all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            augment: Apply augmentation
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"📸 Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            try:
                # Preprocess
                processed = self.preprocess_image(img_path, augment=augment)
                
                # Save
                output_file = output_path / img_path.name
                
                # Convert back to uint8 for saving
                if isinstance(processed, np.ndarray):
                    if processed.dtype == np.float32:
                        processed = (processed * 255).astype(np.uint8)
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_file), processed_bgr)
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
        
        print(f"✅ Processed images saved to {output_dir}")
    
    def enhance_image(self, image):
        """
        Enhance image quality for better detection
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def remove_noise(self, image):
        """Remove noise from image"""
        # Apply bilateral filter
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def extract_roi(self, image, roi_coords=None):
        """
        Extract region of interest (road area)
        
        Args:
            image: Input image
            roi_coords: ROI coordinates [(x1,y1), (x2,y2), ...]
            
        Returns:
            ROI image
        """
        if roi_coords is None:
            # Default: bottom 2/3 of image (typical road area)
            h, w = image.shape[:2]
            roi_coords = np.array([
                [0, h // 3],
                [w, h // 3],
                [w, h],
                [0, h]
            ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_coords], 255)
        
        # Apply mask
        roi = cv2.bitwise_and(image, image, mask=mask)
        
        return roi
    
    def normalize_lighting(self, image):
        """Normalize lighting conditions"""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Equalize Y channel
        y_eq = cv2.equalizeHist(y)
        
        # Merge and convert back
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        normalized = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        
        return normalized


class VideoPreprocessor:
    """Video preprocessing for traffic analysis"""
    
    def __init__(self, target_size=(640, 640)):
        self.target_size = target_size
        self.image_preprocessor = ImagePreprocessor(target_size)
    
    def extract_frames(self, video_path, output_dir, frame_interval=30):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            frame_interval: Extract every Nth frame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎥 Video info:")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Extracting every {frame_interval} frames...")
        
        frame_count = 0
        saved_count = 0
        
        with tqdm(total=total_frames // frame_interval) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save frame
                    output_file = output_path / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(output_file), frame)
                    saved_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        print(f"✅ Extracted {saved_count} frames to {output_dir}")
    
    def create_video_from_frames(self, frames_dir, output_video, fps=30):
        """Create video from frames"""
        frames_path = Path(frames_dir)
        frame_files = sorted(frames_path.glob('*.jpg'))
        
        if not frame_files:
            print("❌ No frames found!")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
        
        print(f"🎬 Creating video from {len(frame_files)} frames...")
        
        for frame_file in tqdm(frame_files):
            frame = cv2.imread(str(frame_file))
            out.write(frame)
        
        out.release()
        print(f"✅ Video saved to {output_video}")


def main():
    """Example usage"""
    
    # Image preprocessing
    preprocessor = ImagePreprocessor(target_size=(640, 640))
    
    # Process training images with augmentation
    if os.path.exists('data/images/train'):
        preprocessor.batch_preprocess(
            'data/images/train/congested',
            'data/processed/train/congested',
            augment=True
        )
        preprocessor.batch_preprocess(
            'data/images/train/free_flow',
            'data/processed/train/free_flow',
            augment=True
        )
    
    # Video preprocessing
    video_preprocessor = VideoPreprocessor()
    
    # Extract frames from videos if available
    video_dir = Path('data/videos')
    if video_dir.exists():
        for video_file in video_dir.glob('*.mp4'):
            output_dir = f'data/processed/frames/{video_file.stem}'
            video_preprocessor.extract_frames(video_file, output_dir, frame_interval=30)
    
    print("\n✅ Preprocessing completed!")


if __name__ == '__main__':
    main()
