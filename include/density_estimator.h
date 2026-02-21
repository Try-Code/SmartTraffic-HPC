#ifndef DENSITY_ESTIMATOR_H
#define DENSITY_ESTIMATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "vehicle_detector.h"

namespace traffic {

enum class CongestionLevel {
    FREE_FLOW,      // 0-20% occupancy
    LIGHT,          // 20-40% occupancy
    MODERATE,       // 40-60% occupancy
    HEAVY,          // 60-80% occupancy
    SEVERE          // 80-100% occupancy
};

struct DensityMetrics {
    int vehicleCount;
    double occupancyRatio;
    double averageVehicleSize;
    CongestionLevel congestionLevel;
    double densityScore;  // vehicles per square meter
};

class DensityEstimator {
public:
    DensityEstimator();
    explicit DensityEstimator(const cv::Size& roiSize);
    
    // Density estimation methods
    DensityMetrics estimateDensity(const cv::Mat& image, const std::vector<Detection>& detections);
    DensityMetrics estimateDensityFromPixels(const cv::Mat& image);
    
    // Grid-based density estimation
    cv::Mat computeDensityMap(const cv::Mat& image, const std::vector<Detection>& detections, int gridSize = 32);
    std::vector<std::vector<int>> computeGridDensity(const std::vector<Detection>& detections, int gridRows, int gridCols);
    
    // Congestion detection
    CongestionLevel detectCongestionLevel(double occupancyRatio);
    bool isCongested(const DensityMetrics& metrics, double threshold = 0.6);
    
    // ROI management
    void setROI(const cv::Rect& roi);
    cv::Rect getROI() const { return regionOfInterest; }
    
    // Visualization
    cv::Mat visualizeDensityMap(const cv::Mat& densityMap);
    cv::Mat drawDensityMetrics(const cv::Mat& image, const DensityMetrics& metrics);
    
    // Calibration
    void calibrateRoadArea(double roadAreaInSquareMeters);
    void setOccupancyThresholds(double light, double moderate, double heavy, double severe);
    
private:
    cv::Rect regionOfInterest;
    cv::Size imageSize;
    double roadArea;  // in square meters
    
    // Thresholds for congestion levels
    double lightThreshold;
    double moderateThreshold;
    double heavyThreshold;
    double severeThreshold;
    
    // Helper methods
    double calculateOccupancyRatio(const std::vector<Detection>& detections);
    double calculateAverageVehicleSize(const std::vector<Detection>& detections);
    std::string congestionLevelToString(CongestionLevel level);
};

} // namespace traffic

#endif // DENSITY_ESTIMATOR_H
