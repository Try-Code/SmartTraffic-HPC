#include "density_estimator.h"
#include <algorithm>
#include <numeric>

namespace traffic {

DensityEstimator::DensityEstimator() 
    : roadArea(1000.0),  // Default 1000 square meters
      lightThreshold(0.2),
      moderateThreshold(0.4),
      heavyThreshold(0.6),
      severeThreshold(0.8) {}

DensityEstimator::DensityEstimator(const cv::Size& roiSize) 
    : DensityEstimator() {
    imageSize = roiSize;
    regionOfInterest = cv::Rect(0, 0, roiSize.width, roiSize.height);
}

DensityMetrics DensityEstimator::estimateDensity(const cv::Mat& image, 
                                                 const std::vector<Detection>& detections) {
    DensityMetrics metrics;
    
    metrics.vehicleCount = detections.size();
    metrics.occupancyRatio = calculateOccupancyRatio(detections);
    metrics.averageVehicleSize = calculateAverageVehicleSize(detections);
    metrics.congestionLevel = detectCongestionLevel(metrics.occupancyRatio);
    metrics.densityScore = metrics.vehicleCount / roadArea;
    
    return metrics;
}

DensityMetrics DensityEstimator::estimateDensityFromPixels(const cv::Mat& image) {
    DensityMetrics metrics;
    
    // Convert to grayscale and threshold
    cv::Mat gray, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    
    // Count non-zero pixels (vehicles)
    int vehiclePixels = cv::countNonZero(binary);
    int totalPixels = image.rows * image.cols;
    
    metrics.occupancyRatio = static_cast<double>(vehiclePixels) / totalPixels;
    metrics.congestionLevel = detectCongestionLevel(metrics.occupancyRatio);
    metrics.vehicleCount = 0;  // Unknown from pixel-based method
    metrics.averageVehicleSize = 0.0;
    metrics.densityScore = metrics.occupancyRatio;
    
    return metrics;
}

cv::Mat DensityEstimator::computeDensityMap(const cv::Mat& image, 
                                           const std::vector<Detection>& detections, 
                                           int gridSize) {
    cv::Mat densityMap = cv::Mat::zeros(image.rows, image.cols, CV_32F);
    
    for (const auto& detection : detections) {
        cv::Point center(detection.boundingBox.x + detection.boundingBox.width / 2,
                        detection.boundingBox.y + detection.boundingBox.height / 2);
        
        // Create Gaussian distribution around vehicle center
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                double distance = std::sqrt(std::pow(x - center.x, 2) + std::pow(y - center.y, 2));
                double sigma = gridSize;
                double value = std::exp(-(distance * distance) / (2 * sigma * sigma));
                densityMap.at<float>(y, x) += value;
            }
        }
    }
    
    // Normalize
    cv::normalize(densityMap, densityMap, 0, 255, cv::NORM_MINMAX);
    densityMap.convertTo(densityMap, CV_8U);
    
    return densityMap;
}

std::vector<std::vector<int>> DensityEstimator::computeGridDensity(
    const std::vector<Detection>& detections, int gridRows, int gridCols) {
    
    std::vector<std::vector<int>> grid(gridRows, std::vector<int>(gridCols, 0));
    
    int cellWidth = regionOfInterest.width / gridCols;
    int cellHeight = regionOfInterest.height / gridRows;
    
    for (const auto& detection : detections) {
        cv::Point center(detection.boundingBox.x + detection.boundingBox.width / 2,
                        detection.boundingBox.y + detection.boundingBox.height / 2);
        
        int gridX = center.x / cellWidth;
        int gridY = center.y / cellHeight;
        
        if (gridX >= 0 && gridX < gridCols && gridY >= 0 && gridY < gridRows) {
            grid[gridY][gridX]++;
        }
    }
    
    return grid;
}

CongestionLevel DensityEstimator::detectCongestionLevel(double occupancyRatio) {
    if (occupancyRatio < lightThreshold) {
        return CongestionLevel::FREE_FLOW;
    } else if (occupancyRatio < moderateThreshold) {
        return CongestionLevel::LIGHT;
    } else if (occupancyRatio < heavyThreshold) {
        return CongestionLevel::MODERATE;
    } else if (occupancyRatio < severeThreshold) {
        return CongestionLevel::HEAVY;
    } else {
        return CongestionLevel::SEVERE;
    }
}

bool DensityEstimator::isCongested(const DensityMetrics& metrics, double threshold) {
    return metrics.occupancyRatio >= threshold;
}

void DensityEstimator::setROI(const cv::Rect& roi) {
    regionOfInterest = roi;
}

cv::Mat DensityEstimator::visualizeDensityMap(const cv::Mat& densityMap) {
    cv::Mat colorMap;
    cv::applyColorMap(densityMap, colorMap, cv::COLORMAP_JET);
    return colorMap;
}

cv::Mat DensityEstimator::drawDensityMetrics(const cv::Mat& image, const DensityMetrics& metrics) {
    cv::Mat result = image.clone();
    
    int y = 30;
    int lineHeight = 30;
    
    // Draw background rectangle
    cv::rectangle(result, cv::Point(10, 10), cv::Point(400, 180), 
                 cv::Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(result, cv::Point(10, 10), cv::Point(400, 180), 
                 cv::Scalar(255, 255, 255), 2);
    
    // Draw metrics
    cv::putText(result, "Vehicle Count: " + std::to_string(metrics.vehicleCount),
               cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    y += lineHeight;
    
    cv::putText(result, "Occupancy: " + std::to_string((int)(metrics.occupancyRatio * 100)) + "%",
               cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    y += lineHeight;
    
    cv::putText(result, "Density: " + std::to_string(metrics.densityScore).substr(0, 5) + " veh/m²",
               cv::Point(20, y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    y += lineHeight;
    
    std::string levelStr = "Level: " + congestionLevelToString(metrics.congestionLevel);
    cv::Scalar levelColor;
    
    switch (metrics.congestionLevel) {
        case CongestionLevel::FREE_FLOW:
            levelColor = cv::Scalar(0, 255, 0);  // Green
            break;
        case CongestionLevel::LIGHT:
            levelColor = cv::Scalar(0, 255, 255);  // Yellow
            break;
        case CongestionLevel::MODERATE:
            levelColor = cv::Scalar(0, 165, 255);  // Orange
            break;
        case CongestionLevel::HEAVY:
            levelColor = cv::Scalar(0, 0, 255);  // Red
            break;
        case CongestionLevel::SEVERE:
            levelColor = cv::Scalar(128, 0, 128);  // Purple
            break;
    }
    
    cv::putText(result, levelStr, cv::Point(20, y), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, levelColor, 2);
    
    return result;
}

void DensityEstimator::calibrateRoadArea(double roadAreaInSquareMeters) {
    roadArea = roadAreaInSquareMeters;
}

void DensityEstimator::setOccupancyThresholds(double light, double moderate, 
                                              double heavy, double severe) {
    lightThreshold = light;
    moderateThreshold = moderate;
    heavyThreshold = heavy;
    severeThreshold = severe;
}

double DensityEstimator::calculateOccupancyRatio(const std::vector<Detection>& detections) {
    if (detections.empty()) return 0.0;
    
    double totalArea = regionOfInterest.width * regionOfInterest.height;
    double occupiedArea = 0.0;
    
    for (const auto& detection : detections) {
        occupiedArea += detection.boundingBox.width * detection.boundingBox.height;
    }
    
    return occupiedArea / totalArea;
}

double DensityEstimator::calculateAverageVehicleSize(const std::vector<Detection>& detections) {
    if (detections.empty()) return 0.0;
    
    double totalSize = 0.0;
    for (const auto& detection : detections) {
        totalSize += detection.boundingBox.width * detection.boundingBox.height;
    }
    
    return totalSize / detections.size();
}

std::string DensityEstimator::congestionLevelToString(CongestionLevel level) {
    switch (level) {
        case CongestionLevel::FREE_FLOW: return "FREE FLOW";
        case CongestionLevel::LIGHT: return "LIGHT";
        case CongestionLevel::MODERATE: return "MODERATE";
        case CongestionLevel::HEAVY: return "HEAVY";
        case CongestionLevel::SEVERE: return "SEVERE";
        default: return "UNKNOWN";
    }
}

} // namespace traffic
