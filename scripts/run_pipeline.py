"""
Complete Pipeline Runner
Runs the entire traffic analysis pipeline from preprocessing to analysis
"""

import os
import sys
from pathlib import Path
import argparse
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))

from image_preprocessor import ImagePreprocessor, VideoPreprocessor
from train_yolo import TrafficYOLOTrainer
from realtime_processor import RealtimeTrafficProcessor


class TrafficPipeline:
    """Complete traffic analysis pipeline"""
    
    def __init__(self, config_path='config/processing_config.json'):
        self.config = self.load_config(config_path)
        self.preprocessor = ImagePreprocessor()
        self.video_preprocessor = VideoPreprocessor()
        
    def load_config(self, config_path):
        """Load configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            return {}
    
    def run_preprocessing(self, input_dir='data/images/train', 
                         output_dir='data/processed/train'):
        """Run image preprocessing"""
        print("\n" + "="*60)
        print("STEP 1: Image Preprocessing")
        print("="*60)
        
        self.preprocessor.batch_preprocess(
            input_dir=input_dir,
            output_dir=output_dir,
            augment=True
        )
    
    def run_training(self, epochs=100, model_size='n'):
        """Run model training"""
        print("\n" + "="*60)
        print("STEP 2: Model Training")
        print("="*60)
        
        trainer = TrafficYOLOTrainer(model_size=model_size)
        
        # Create data config
        trainer.create_data_yaml(
            train_path='data/images/train',
            val_path='data/images/validation',
            test_path='data/images/test'
        )
        
        # Train
        results = trainer.train(
            epochs=epochs,
            imgsz=640,
            batch=16,
            device='cpu'  # Change to 'cuda' if GPU available
        )
        
        # Validate
        metrics = trainer.validate()
        
        # Export
        trainer.export(format='onnx')
        
        return results, metrics
    
    def run_inference(self, source, output_path=None):
        """Run inference on video/images"""
        print("\n" + "="*60)
        print("STEP 3: Inference")
        print("="*60)
        
        processor = RealtimeTrafficProcessor(
            model_path='models/yolov8n.pt',
            conf_threshold=0.3
        )
        
        if Path(source).is_file():
            # Video file
            processor.process_video(
                video_source=source,
                output_path=output_path,
                display=True,
                save_stats=True
            )
        elif Path(source).is_dir():
            # Image directory
            output_dir = Path('output/processed_images')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in Path(source).glob('*.jpg'):
                output_file = output_dir / f"processed_{img_file.name}"
                processor.process_image(img_file, output_file)
        else:
            print(f"❌ Invalid source: {source}")
    
    def run_full_pipeline(self, preprocess=True, train=False, inference=True):
        """Run complete pipeline"""
        print("\n" + "="*60)
        print("🚀 SMART TRAFFIC ANALYSIS PIPELINE")
        print("="*60)
        
        # Step 1: Preprocessing
        if preprocess:
            if Path('data/images/train').exists():
                self.run_preprocessing()
            else:
                print("⚠️  Training images not found, skipping preprocessing")
        
        # Step 2: Training
        if train:
            self.run_training(epochs=50, model_size='n')
        
        # Step 3: Inference
        if inference:
            # Try to find test data
            if Path('data/videos').exists():
                videos = list(Path('data/videos').glob('*.mp4'))
                if videos:
                    self.run_inference(
                        source=str(videos[0]),
                        output_path='output/processed_video.mp4'
                    )
            elif Path('data/images/test').exists():
                self.run_inference(source='data/images/test')
            else:
                print("⚠️  No test data found")
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED!")
        print("="*60)
        print("\n📁 Check the following directories for results:")
        print("   - output/visualizations/  - Annotated images")
        print("   - output/results/         - Detection results")
        print("   - output/metrics/         - Performance metrics")


def main():
    parser = argparse.ArgumentParser(description='Traffic Analysis Pipeline')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['preprocess', 'train', 'inference', 'full'],
                       help='Pipeline mode')
    parser.add_argument('--source', type=str, default=None,
                       help='Source for inference (video file or image directory)')
    parser.add_argument('--output', type=str, default='output/processed_video.mp4',
                       help='Output path for processed video')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrafficPipeline()
    
    # Run based on mode
    if args.mode == 'preprocess':
        pipeline.run_preprocessing()
    
    elif args.mode == 'train':
        pipeline.run_training(epochs=args.epochs, model_size=args.model_size)
    
    elif args.mode == 'inference':
        if args.source is None:
            print("❌ --source is required for inference mode")
            sys.exit(1)
        pipeline.run_inference(source=args.source, output_path=args.output)
    
    elif args.mode == 'full':
        pipeline.run_full_pipeline(
            preprocess=True,
            train=False,  # Set to True to include training
            inference=True
        )


if __name__ == '__main__':
    main()
