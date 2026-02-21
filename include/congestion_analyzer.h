#ifndef CONGESTION_ANALYZER_H
#define CONGESTION_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "vehicle_detector.h"
#include "density_estimator.h"
#include "speed_tracker.h"

namespace traffic {

enum class TrafficState {
    FREE_FLOW,
    SYNCHRONIZED,
    WIDE_MOVING_JAM,
    STOPPED
};

struct CongestionMetrics {
    CongestionLevel congestionLevel;
    TrafficState trafficState;
    double densityScore;
    double averageSpeed;
    int vehicleCount;
    double occupancyRatio;
    double flowRate;  // vehicles per minute
    double timeToTraverse;  // estimated time in seconds
    std::string recommendation;
};

struct HistoricalData {
    std::vector<double> densityHistory;
    std::vector<double> speedHistory;
    std::vector<int> vehicleCountHistory;
    std::vector<CongestionLevel> congestionHistory;
    int maxHistorySize;
};

class CongestionAnalyzer {
public:
    CongestionAnalyzer();
    explicit CongestionAnalyzer(const cv::Size& frameSize);
    
    // Main analysis method
    CongestionMetrics analyzeCongestion(
        const cv::Mat& frame,
        const std::vector<Detection>& detections,
        const DensityMetrics& densityMetrics,
        const SpeedMetrics& speedMetrics
    );
    
    // Traffic state detection
    TrafficState detectTrafficState(double density, double speed);
    std::string trafficStateToString(TrafficState state);
    
    // Flow rate calculation
    double calculateFlowRate(int vehicleCount, double timeWindow);
    double estimateTimeToTraverse(double distance, double averageSpeed);
    
    // Prediction
    CongestionLevel predictCongestion(int futureFrames = 30);
    bool isCongestionIncreasing();
    bool isCongestionDecreasing();
    
    // Historical analysis
    void updateHistory(const CongestionMetrics& metrics);
    HistoricalData getHistoricalData() const { return history; }
    void clearHistory();
    
    // Recommendations
    std::string generateRecommendation(const CongestionMetrics& metrics);
    bool shouldRerouteTraffic(const CongestionMetrics& metrics);
    bool shouldAdjustSignalTiming(const CongestionMetrics& metrics);
    
    // Visualization
    cv::Mat visualizeCongestion(const cv::Mat& frame, const CongestionMetrics& metrics);
    cv::Mat drawCongestionHeatmap(const cv::Mat& frame, const std::vector<Detection>& detections);
    cv::Mat drawHistoricalTrends(int width = 800, int height = 400);
    
    // Configuration
    void setRoadLength(double lengthInMeters);
    void setHistorySize(int maxSize);
    void setCongestionThresholds(double light, double moderate, double heavy);
    
    // Statistics
    std::map<CongestionLevel, int> getCongestionStatistics();
    double getAverageCongestionLevel();
    
private:
    cv::Size frameSize;
    double roadLength;  // in meters
    HistoricalData history;
    
    // Thresholds
    double lightCongestionThreshold;
    double moderateCongestionThreshold;
    double heavyCongestionThreshold;
    
    // Time tracking
    std::chrono::steady_clock::time_point lastUpdateTime;
    int frameCounter;
    
    // Helper methods
    cv::Scalar getCongestionColor(CongestionLevel level);
    cv::Scalar getTrafficStateColor(TrafficState state);
    double calculateTrend(const std::vector<double>& data, int windowSize = 10);
    void addToHistory(std::vector<double>& history, double value, int maxSize);
};

} // namespace traffic

#endif // CONGESTION_ANALYZER_H
