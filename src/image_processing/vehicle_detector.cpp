#include "vehicle_detector.h"
#include "image_utils.h"
#include <iostream>
#include <fstream>

namespace traffic {

VehicleDetector::VehicleDetector() : detectionCount(0) {
    // Initialize with common vehicle classes
    classNames = {"background", "aeroplane", "bicycle", "bird", "boat",
                  "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                  "sofa", "train", "tvmonitor"};
}

VehicleDetector::VehicleDetector(const std::string& modelPath) : VehicleDetector() {
    loadModel(modelPath);
}

VehicleDetector::~VehicleDetector() {}

bool VehicleDetector::loadModel(const std::string& modelPath, const std::string& configPath) {
    try {
        if (configPath.empty()) {
            net = cv::dnn::readNet(modelPath);
        } else {
            net = cv::dnn::readNet(modelPath, configPath);
        }
        
        // Set backend and target
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::cout << "Model loaded successfully from: " << modelPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

bool VehicleDetector::loadYOLOModel(const std::string& weightsPath, const std::string& configPath) {
    try {
        net = cv::dnn::readNetFromDarknet(configPath, weightsPath);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        outputLayerNames = getOutputsNames();
        
        // Load COCO class names
        classNames = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"};
        
        std::cout << "YOLO model loaded successfully" << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading YOLO model: " << e.what() << std::endl;
        return false;
    }
}

bool VehicleDetector::loadCaffeModel(const std::string& prototxt, const std::string& caffeModel) {
    try {
        net = cv::dnn::readNetFromCaffe(prototxt, caffeModel);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        
        std::cout << "Caffe model loaded successfully" << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading Caffe model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<Detection> VehicleDetector::detect(const cv::Mat& image, 
                                               float confidenceThreshold, 
                                               float nmsThreshold) {
    std::vector<Detection> detections;
    
    if (net.empty()) {
        std::cerr << "Model not loaded!" << std::endl;
        return detections;
    }
    
    // Create blob from image
    cv::Mat blob = cv::dnn::blobFromImage(image, 1/255.0, cv::Size(416, 416), 
                                          cv::Scalar(0, 0, 0), true, false);
    
    net.setInput(blob);
    
    // Forward pass
    std::vector<cv::Mat> outs;
    net.forward(outs, outputLayerNames);
    
    // Post-process detections
    cv::Mat frame = image.clone();
    postprocess(frame, outs, detections, confidenceThreshold, nmsThreshold);
    
    detectionCount = detections.size();
    return detections;
}

std::vector<Detection> VehicleDetector::detectVehicles(const cv::Mat& image) {
    std::vector<Detection> allDetections = detect(image);
    std::vector<Detection> vehicleDetections;
    
    // Filter only vehicle classes (car, bus, truck, motorbike, bicycle)
    std::vector<std::string> vehicleClasses = {"car", "bus", "truck", "motorbike", "bicycle", "train"};
    
    for (const auto& detection : allDetections) {
        for (const auto& vehicleClass : vehicleClasses) {
            if (detection.className == vehicleClass) {
                vehicleDetections.push_back(detection);
                break;
            }
        }
    }
    
    return vehicleDetections;
}

void VehicleDetector::postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs,
                                  std::vector<Detection>& detections,
                                  float confidenceThreshold, float nmsThreshold) {
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            
            if (confidence > confidenceThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    
    // Non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Detection detection;
        detection.boundingBox = boxes[idx];
        detection.confidence = confidences[idx];
        detection.classId = classIds[idx];
        detection.className = (classIds[idx] < classNames.size()) ? 
                             classNames[classIds[idx]] : "unknown";
        detections.push_back(detection);
    }
}

std::vector<std::string> VehicleDetector::getOutputsNames() {
    std::vector<std::string> names;
    if (net.empty()) return names;
    
    std::vector<int> outLayers = net.getUnconnectedOutLayers();
    std::vector<std::string> layersNames = net.getLayerNames();
    
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
        names[i] = layersNames[outLayers[i] - 1];
    }
    
    return names;
}

cv::Mat VehicleDetector::backgroundSubtraction(const cv::Mat& image) {
    if (backgroundSubtractor.empty()) {
        initializeBackgroundSubtractor();
    }
    
    cv::Mat fgMask;
    backgroundSubtractor->apply(image, fgMask);
    
    return fgMask;
}

void VehicleDetector::initializeBackgroundSubtractor(int history, double varThreshold, bool detectShadows) {
    backgroundSubtractor = cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
}

bool VehicleDetector::loadCascadeClassifier(const std::string& cascadePath) {
    return cascadeClassifier.load(cascadePath);
}

std::vector<cv::Rect> VehicleDetector::detectWithCascade(const cv::Mat& image) {
    std::vector<cv::Rect> detections;
    
    if (cascadeClassifier.empty()) {
        std::cerr << "Cascade classifier not loaded!" << std::endl;
        return detections;
    }
    
    cv::Mat gray = ImageUtils::convertToGrayscale(image);
    cascadeClassifier.detectMultiScale(gray, detections, 1.1, 3, 0, cv::Size(30, 30));
    
    return detections;
}

cv::Mat VehicleDetector::drawDetections(const cv::Mat& image, const std::vector<Detection>& detections) {
    cv::Mat result = image.clone();
    
    for (const auto& detection : detections) {
        // Draw bounding box
        cv::rectangle(result, detection.boundingBox, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = detection.className + ": " + 
                          std::to_string((int)(detection.confidence * 100)) + "%";
        
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int top = std::max(detection.boundingBox.y, labelSize.height);
        cv::rectangle(result, 
                     cv::Point(detection.boundingBox.x, top - labelSize.height),
                     cv::Point(detection.boundingBox.x + labelSize.width, top + baseLine),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        
        cv::putText(result, label, 
                   cv::Point(detection.boundingBox.x, top),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    return result;
}

} // namespace traffic
