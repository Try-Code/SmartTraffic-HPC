"""
Real-time Traffic Video Processing
Processes video streams for vehicle detection and congestion analysis
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
from pathlib import Path
import json


class RealtimeTrafficProcessor:
    """Real-time traffic video processor"""
    
    def __init__(self, model_path='models/yolov8n.pt', conf_threshold=0.25):
        """
        Initialize processor
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Vehicle classes (COCO dataset)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
        
        # Metrics tracking
        self.frame_times = deque(maxlen=30)
        self.vehicle_counts = deque(maxlen=100)
        
    def process_video(self, video_source, output_path=None, display=True, save_stats=True):
        """
        Process video stream
        
        Args:
            video_source: Video file path, camera index, or RTSP URL
            output_path: Output video path (optional)
            display: Display processed video
            save_stats: Save statistics to file
        """
        # Open video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📹 Video properties:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        
        # Video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Statistics
        stats = {
            'frames_processed': 0,
            'total_vehicles': 0,
            'avg_vehicles_per_frame': 0,
            'congestion_events': 0,
            'processing_fps': 0
        }
        
        frame_count = 0
        
        print("🚀 Starting processing... Press 'q' to quit")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model.predict(
                    frame,
                    conf=self.conf_threshold,
                    classes=list(self.vehicle_classes.keys()),
                    verbose=False
                )
                
                # Process results
                processed_frame, metrics = self.process_frame(frame, results[0])
                
                # Update statistics
                frame_count += 1
                self.vehicle_counts.append(metrics['vehicle_count'])
                stats['frames_processed'] = frame_count
                stats['total_vehicles'] += metrics['vehicle_count']
                
                # Calculate FPS
                process_time = time.time() - start_time
                self.frame_times.append(process_time)
                current_fps = 1.0 / np.mean(self.frame_times)
                stats['processing_fps'] = current_fps
                
                # Draw FPS
                cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display
                if display:
                    cv2.imshow('Traffic Monitoring', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save
                if out:
                    out.write(processed_frame)
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    avg_vehicles = np.mean(self.vehicle_counts)
                    print(f"Frame {frame_count}: {metrics['vehicle_count']} vehicles, "
                          f"Avg: {avg_vehicles:.1f}, FPS: {current_fps:.1f}")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Calculate final statistics
            if self.vehicle_counts:
                stats['avg_vehicles_per_frame'] = np.mean(self.vehicle_counts)
                stats['congestion_events'] = sum(1 for count in self.vehicle_counts if count > 20)
            
            # Save statistics
            if save_stats:
                stats_file = Path(output_path).with_suffix('.json') if output_path else 'output/stats.json'
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"📊 Statistics saved to {stats_file}")
            
            print("\n✅ Processing completed!")
            print(f"   Total frames: {stats['frames_processed']}")
            print(f"   Average vehicles: {stats['avg_vehicles_per_frame']:.1f}")
            print(f"   Congestion events: {stats['congestion_events']}")
            print(f"   Average FPS: {stats['processing_fps']:.1f}")
    
    def process_frame(self, frame, results):
        """
        Process single frame
        
        Args:
            frame: Input frame
            results: YOLO detection results
            
        Returns:
            Processed frame and metrics
        """
        processed = frame.copy()
        
        # Get detections
        boxes = results.boxes
        vehicle_count = 0
        
        # Draw detections
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Get class name
            class_name = self.vehicle_classes.get(cls, 'unknown')
            
            # Draw bounding box
            color = self.get_color_for_class(cls)
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(processed, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(processed, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            vehicle_count += 1
        
        # Calculate congestion level
        congestion_level = self.calculate_congestion(vehicle_count)
        
        # Draw info panel
        self.draw_info_panel(processed, vehicle_count, congestion_level)
        
        metrics = {
            'vehicle_count': vehicle_count,
            'congestion_level': congestion_level
        }
        
        return processed, metrics
    
    def calculate_congestion(self, vehicle_count):
        """Calculate congestion level based on vehicle count"""
        if vehicle_count < 5:
            return 'FREE_FLOW'
        elif vehicle_count < 10:
            return 'LIGHT'
        elif vehicle_count < 20:
            return 'MODERATE'
        elif vehicle_count < 30:
            return 'HEAVY'
        else:
            return 'SEVERE'
    
    def draw_info_panel(self, frame, vehicle_count, congestion_level):
        """Draw information panel on frame"""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 150), (350, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        y = h - 120
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y += 40
        # Color based on congestion
        color_map = {
            'FREE_FLOW': (0, 255, 0),
            'LIGHT': (0, 255, 255),
            'MODERATE': (0, 165, 255),
            'HEAVY': (0, 0, 255),
            'SEVERE': (128, 0, 128)
        }
        color = color_map.get(congestion_level, (255, 255, 255))
        cv2.putText(frame, f"Status: {congestion_level}", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def get_color_for_class(self, cls):
        """Get color for vehicle class"""
        colors = {
            2: (0, 255, 0),      # car - green
            3: (255, 0, 0),      # motorcycle - blue
            5: (0, 0, 255),      # bus - red
            7: (255, 255, 0),    # truck - cyan
            1: (255, 0, 255)     # bicycle - magenta
        }
        return colors.get(cls, (255, 255, 255))
    
    def process_image(self, image_path, output_path=None):
        """Process single image"""
        # Read image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=list(self.vehicle_classes.keys()),
            verbose=False
        )
        
        # Process
        processed, metrics = self.process_frame(frame, results[0])
        
        # Save
        if output_path:
            cv2.imwrite(output_path, processed)
            print(f"✅ Processed image saved to {output_path}")
        
        return processed, metrics


def main():
    """Example usage"""
    
    # Initialize processor
    processor = RealtimeTrafficProcessor(
        model_path='models/yolov8n.pt',
        conf_threshold=0.3
    )
    
    # Process video file
    video_path = 'data/videos/traffic_sample.mp4'
    if Path(video_path).exists():
        processor.process_video(
            video_source=video_path,
            output_path='output/processed_video.mp4',
            display=True,
            save_stats=True
        )
    else:
        print(f"⚠️  Video not found: {video_path}")
        print("   Place a video file in data/videos/ or use webcam (source=0)")
        
        # Try webcam
        try:
            processor.process_video(
                video_source=0,  # Webcam
                output_path='output/webcam_output.mp4',
                display=True
            )
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Process images
    image_dir = Path('data/images/test')
    if image_dir.exists():
        output_dir = Path('output/processed_images')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in image_dir.glob('*.jpg'):
            output_file = output_dir / f"processed_{img_file.name}"
            processor.process_image(img_file, output_file)


if __name__ == '__main__':
    main()
