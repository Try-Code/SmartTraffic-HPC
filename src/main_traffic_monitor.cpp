#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include "vehicle_detector.h"
#include "density_estimator.h"
#include "speed_tracker.h"
#include "congestion_analyzer.h"
#include "visualization.h"
#include "image_utils.h"

using namespace traffic;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path or camera_index>" << std::endl;
        std::cout << "Example: " << argv[0] << " traffic_video.mp4" << std::endl;
        std::cout << "Example: " << argv[0] << " 0  (for webcam)" << std::endl;
        return -1;
    }

    // Initialize components
    std::cout << "Initializing traffic monitoring system..." << std::endl;
    
    VehicleDetector detector;
    DensityEstimator densityEstimator;
    SpeedTracker speedTracker(10.0, 30.0);  // 10 pixels per meter, 30 FPS
    CongestionAnalyzer congestionAnalyzer;
    
    // Load YOLO model (you'll need to provide the model path)
    std::string modelPath = "models/yolov8n.onnx";
    std::string configPath = "";
    
    std::cout << "Loading detection model..." << std::endl;
    if (!detector.loadModel(modelPath, configPath)) {
        std::cerr << "Warning: Could not load YOLO model. Using background subtraction." << std::endl;
        detector.initializeBackgroundSubtractor();
    }
    
    // Open video source
    cv::VideoCapture cap;
    std::string input = argv[1];
    
    // Check if input is a number (camera index)
    bool isCamera = false;
    try {
        int cameraIndex = std::stoi(input);
        cap.open(cameraIndex);
        isCamera = true;
        std::cout << "Opening camera " << cameraIndex << "..." << std::endl;
    } catch (...) {
        cap.open(input);
        std::cout << "Opening video file: " << input << "..." << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source!" << std::endl;
        return -1;
    }
    
    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps == 0) fps = 30.0;
    
    std::cout << "Video properties: " << frameWidth << "x" << frameHeight << " @ " << fps << " FPS" << std::endl;
    
    // Set up video writer (optional)
    cv::VideoWriter writer;
    bool saveVideo = false;
    if (argc > 2) {
        std::string outputPath = argv[2];
        writer.open(outputPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                   fps, cv::Size(frameWidth, frameHeight));
        if (writer.isOpened()) {
            saveVideo = true;
            std::cout << "Saving output to: " << outputPath << std::endl;
        }
    }
    
    // Configure components
    speedTracker.setFPS(fps);
    congestionAnalyzer.setRoadLength(100.0);  // 100 meters road segment
    
    // Processing loop
    std::cout << "\nStarting traffic monitoring..." << std::endl;
    std::cout << "Press 'q' to quit, 's' to save screenshot, 'p' to pause" << std::endl;
    
    cv::Mat frame;
    int frameCount = 0;
    bool paused = false;
    
    auto startTime = std::chrono::steady_clock::now();
    
    while (true) {
        if (!paused) {
            cap >> frame;
            
            if (frame.empty()) {
                if (isCamera) {
                    std::cerr << "Error: Lost camera connection!" << std::endl;
                    break;
                } else {
                    std::cout << "End of video reached." << std::endl;
                    break;
                }
            }
            
            frameCount++;
            
            // Detect vehicles
            std::vector<Detection> detections = detector.detectVehicles(frame);
            
            // Estimate density
            DensityMetrics densityMetrics = densityEstimator.estimateDensity(frame, detections);
            
            // Track speed
            speedTracker.update(frame, detections);
            SpeedMetrics speedMetrics = speedTracker.getSpeedMetrics();
            
            // Analyze congestion
            CongestionMetrics congestionMetrics = congestionAnalyzer.analyzeCongestion(
                frame, detections, densityMetrics, speedMetrics);
            
            // Visualize results
            cv::Mat visualized = frame.clone();
            
            // Draw detections
            Visualization::drawDetections(visualized, detections);
            
            // Draw speed tracks
            visualized = speedTracker.drawSpeedInfo(visualized);
            
            // Draw congestion banner
            Visualization::drawCongestionBanner(visualized, congestionMetrics.congestionLevel,
                "Vehicles: " + std::to_string(congestionMetrics.vehicleCount) + 
                " | Avg Speed: " + std::to_string(static_cast<int>(congestionMetrics.averageSpeed)) + " km/h");
            
            // Draw metrics panel
            std::map<std::string, std::string> metrics;
            metrics["Frame"] = std::to_string(frameCount);
            metrics["Vehicles"] = std::to_string(congestionMetrics.vehicleCount);
            metrics["Avg Speed"] = std::to_string(static_cast<int>(congestionMetrics.averageSpeed)) + " km/h";
            metrics["Occupancy"] = std::to_string(static_cast<int>(congestionMetrics.occupancyRatio * 100)) + "%";
            metrics["Flow Rate"] = std::to_string(static_cast<int>(congestionMetrics.flowRate)) + " veh/min";
            metrics["State"] = congestionAnalyzer.trafficStateToString(congestionMetrics.trafficState);
            
            Visualization::drawMetricsPanel(visualized, metrics, cv::Point(frameWidth - 350, 120));
            
            // Add timestamp
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
            std::string timestamp = "Time: " + std::to_string(elapsed) + "s";
            Visualization::addTimestamp(visualized, timestamp);
            
            // Display
            cv::imshow("Traffic Monitoring System", visualized);
            
            // Save video
            if (saveVideo) {
                writer.write(visualized);
            }
            
            // Print statistics every 30 frames
            if (frameCount % 30 == 0) {
                std::cout << "\n=== Frame " << frameCount << " ===" << std::endl;
                std::cout << "Vehicles: " << congestionMetrics.vehicleCount << std::endl;
                std::cout << "Avg Speed: " << congestionMetrics.averageSpeed << " km/h" << std::endl;
                std::cout << "Congestion: " << static_cast<int>(congestionMetrics.congestionLevel) << std::endl;
                std::cout << "Recommendation: " << congestionMetrics.recommendation << std::endl;
            }
        }
        
        // Handle keyboard input
        int key = cv::waitKey(isCamera ? 1 : 30);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            std::cout << "\nQuitting..." << std::endl;
            break;
        } else if (key == 'p') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << std::endl;
        } else if (key == 's') {
            std::string screenshotPath = "screenshot_" + std::to_string(frameCount) + ".jpg";
            cv::imwrite(screenshotPath, frame);
            std::cout << "Screenshot saved: " << screenshotPath << std::endl;
        }
    }
    
    // Cleanup
    cap.release();
    if (saveVideo) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    // Print final statistics
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total frames processed: " << frameCount << std::endl;
    
    auto congestionStats = congestionAnalyzer.getCongestionStatistics();
    std::cout << "\nCongestion Distribution:" << std::endl;
    for (const auto& stat : congestionStats) {
        std::cout << "  Level " << static_cast<int>(stat.first) << ": " 
                 << stat.second << " frames (" 
                 << (100.0 * stat.second / frameCount) << "%)" << std::endl;
    }
    
    std::cout << "\nTraffic monitoring completed successfully!" << std::endl;
    
    return 0;
}
