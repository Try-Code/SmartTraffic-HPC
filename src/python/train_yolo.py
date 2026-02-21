"""
YOLOv8 Training Script for Traffic Vehicle Detection
This script trains a YOLOv8 model on traffic images
"""

import os
import yaml
from ultralytics import YOLO
import torch
from pathlib import Path

class TrafficYOLOTrainer:
    def __init__(self, model_size='n', data_yaml='config/traffic_data.yaml'):
        """
        Initialize YOLO trainer
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
            data_yaml: Path to data configuration file
        """
        self.model_size = model_size
        self.data_yaml = data_yaml
        self.model = None
        
    def create_data_yaml(self, train_path, val_path, test_path=None):
        """Create YOLO data configuration file"""
        
        # Vehicle classes for traffic monitoring
        classes = {
            0: 'car',
            1: 'bus',
            2: 'truck',
            3: 'motorbike',
            4: 'bicycle'
        }
        
        data_config = {
            'path': str(Path.cwd()),
            'train': train_path,
            'val': val_path,
            'test': test_path if test_path else val_path,
            'nc': len(classes),  # number of classes
            'names': list(classes.values())
        }
        
        # Save configuration
        os.makedirs(os.path.dirname(self.data_yaml), exist_ok=True)
        with open(self.data_yaml, 'w') as f:
            yaml.dump(data_config, f)
        
        print(f"✅ Data configuration saved to {self.data_yaml}")
        return data_config
    
    def train(self, epochs=100, imgsz=640, batch=16, device='cpu', **kwargs):
        """
        Train YOLO model
        
        Args:
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            device: 'cpu', 'cuda', or device number
            **kwargs: Additional training arguments
        """
        
        # Load pretrained model
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)
        
        print(f"🚀 Starting training with YOLOv8{self.model_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch}")
        print(f"   Device: {device}")
        
        # Train the model
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='output/yolo_training',
            name='traffic_detector',
            patience=50,  # Early stopping patience
            save=True,
            plots=True,
            **kwargs
        )
        
        print("✅ Training completed!")
        return results
    
    def validate(self):
        """Validate the trained model"""
        if self.model is None:
            print("❌ No model loaded. Train or load a model first.")
            return None
        
        print("📊 Validating model...")
        metrics = self.model.val()
        
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        
        return metrics
    
    def export(self, format='onnx'):
        """
        Export model to different formats
        
        Args:
            format: 'onnx', 'torchscript', 'tflite', 'saved_model', etc.
        """
        if self.model is None:
            print("❌ No model loaded.")
            return None
        
        print(f"📦 Exporting model to {format}...")
        path = self.model.export(format=format)
        print(f"✅ Model exported to {path}")
        return path
    
    def predict(self, source, save=True, conf=0.25):
        """
        Run inference on images/videos
        
        Args:
            source: Path to image, video, or directory
            save: Save results
            conf: Confidence threshold
        """
        if self.model is None:
            # Load best trained model
            self.model = YOLO('output/yolo_training/traffic_detector/weights/best.pt')
        
        print(f"🔍 Running inference on {source}")
        results = self.model.predict(
            source=source,
            save=save,
            conf=conf,
            project='output/predictions',
            name='traffic_detection'
        )
        
        return results


def main():
    """Main training pipeline"""
    
    # Initialize trainer
    trainer = TrafficYOLOTrainer(model_size='n')  # Use nano model for faster training
    
    # Create data configuration
    trainer.create_data_yaml(
        train_path='data/images/train',
        val_path='data/images/validation',
        test_path='data/images/test'
    )
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Train model
    results = trainer.train(
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        optimizer='Adam',
        lr0=0.001,
        augment=True
    )
    
    # Validate
    metrics = trainer.validate()
    
    # Export to ONNX for deployment
    trainer.export(format='onnx')
    
    print("\n🎉 Training pipeline completed!")
    print("📁 Check 'output/yolo_training' for results")


if __name__ == '__main__':
    main()
