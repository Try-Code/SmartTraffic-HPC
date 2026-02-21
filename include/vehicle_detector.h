#ifndef VEHICLE_DETECTOR_H
#define VEHICLE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

namespace traffic {

struct Detection {
    cv::Rect boundingBox;
    float confidence;
    int classId;
    std::string className;
};

class VehicleDetector {
public:
    VehicleDetector();
    explicit VehicleDetector(const std::string& modelPath);
    ~VehicleDetector();
    
    // Model loading
    bool loadModel(const std::string& modelPath, const std::string& configPath = "");
    bool loadYOLOModel(const std::string& weightsPath, const std::string& configPath);
    bool loadCaffeModel(const std::string& prototxt, const std::string& caffeModel);
    
    // Detection methods
    std::vector<Detection> detect(const cv::Mat& image, float confidenceThreshold = 0.5, float nmsThreshold = 0.4);
    std::vector<Detection> detectVehicles(const cv::Mat& image);
    
    // Background subtraction methods
    cv::Mat backgroundSubtraction(const cv::Mat& image);
    void initializeBackgroundSubtractor(int history = 500, double varThreshold = 16, bool detectShadows = true);
    
    // Cascade classifier (for simple detection)
    bool loadCascadeClassifier(const std::string& cascadePath);
    std::vector<cv::Rect> detectWithCascade(const cv::Mat& image);
    
    // Visualization
    cv::Mat drawDetections(const cv::Mat& image, const std::vector<Detection>& detections);
    
    // Getters
    int getDetectionCount() const { return detectionCount; }
    std::vector<std::string> getClassNames() const { return classNames; }
    
private:
    cv::dnn::Net net;
    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor;
    cv::CascadeClassifier cascadeClassifier;
    
    std::vector<std::string> classNames;
    std::vector<std::string> outputLayerNames;
    int detectionCount;
    
    // Helper methods
    void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, 
                    std::vector<Detection>& detections, 
                    float confidenceThreshold, float nmsThreshold);
    std::vector<std::string> getOutputsNames();
};

} // namespace traffic

#endif // VEHICLE_DETECTOR_H
