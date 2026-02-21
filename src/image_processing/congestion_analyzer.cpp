#include "congestion_analyzer.h"
#include "visualization.h"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>

namespace traffic {

CongestionAnalyzer::CongestionAnalyzer()
    : frameSize(cv::Size(1920, 1080)), roadLength(100.0), frameCounter(0),
      lightCongestionThreshold(0.3), moderateCongestionThreshold(0.5), 
      heavyCongestionThreshold(0.7) {
    history.maxHistorySize = 300;  // Store last 10 seconds at 30 FPS
    lastUpdateTime = std::chrono::steady_clock::now();
}

CongestionAnalyzer::CongestionAnalyzer(const cv::Size& frameSize)
    : frameSize(frameSize), roadLength(100.0), frameCounter(0),
      lightCongestionThreshold(0.3), moderateCongestionThreshold(0.5),
      heavyCongestionThreshold(0.7) {
    history.maxHistorySize = 300;
    lastUpdateTime = std::chrono::steady_clock::now();
}

CongestionMetrics CongestionAnalyzer::analyzeCongestion(
    const cv::Mat& frame,
    const std::vector<Detection>& detections,
    const DensityMetrics& densityMetrics,
    const SpeedMetrics& speedMetrics) {
    
    CongestionMetrics metrics;
    
    // Basic metrics
    metrics.congestionLevel = densityMetrics.congestionLevel;
    metrics.densityScore = densityMetrics.densityScore;
    metrics.vehicleCount = densityMetrics.vehicleCount;
    metrics.occupancyRatio = densityMetrics.occupancyRatio;
    metrics.averageSpeed = speedMetrics.averageSpeed;
    
    // Detect traffic state
    metrics.trafficState = detectTrafficState(metrics.densityScore, metrics.averageSpeed);
    
    // Calculate flow rate (vehicles per minute)
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastUpdateTime).count();
    if (elapsed > 0) {
        metrics.flowRate = (metrics.vehicleCount * 60.0) / elapsed;
    } else {
        metrics.flowRate = 0.0;
    }
    
    // Estimate time to traverse
    metrics.timeToTraverse = estimateTimeToTraverse(roadLength, metrics.averageSpeed);
    
    // Generate recommendation
    metrics.recommendation = generateRecommendation(metrics);
    
    // Update history
    updateHistory(metrics);
    
    frameCounter++;
    
    return metrics;
}

TrafficState CongestionAnalyzer::detectTrafficState(double density, double speed) {
    // Based on three-phase traffic theory
    if (density < 20 && speed > 60) {
        return TrafficState::FREE_FLOW;
    } else if (density >= 20 && density < 50 && speed > 30) {
        return TrafficState::SYNCHRONIZED;
    } else if (density >= 50 && speed > 10) {
        return TrafficState::WIDE_MOVING_JAM;
    } else {
        return TrafficState::STOPPED;
    }
}

std::string CongestionAnalyzer::trafficStateToString(TrafficState state) {
    switch (state) {
        case TrafficState::FREE_FLOW:
            return "Free Flow";
        case TrafficState::SYNCHRONIZED:
            return "Synchronized Flow";
        case TrafficState::WIDE_MOVING_JAM:
            return "Wide Moving Jam";
        case TrafficState::STOPPED:
            return "Stopped";
        default:
            return "Unknown";
    }
}

double CongestionAnalyzer::calculateFlowRate(int vehicleCount, double timeWindow) {
    if (timeWindow <= 0) return 0.0;
    return (vehicleCount / timeWindow) * 60.0;  // vehicles per minute
}

double CongestionAnalyzer::estimateTimeToTraverse(double distance, double averageSpeed) {
    if (averageSpeed <= 0) return -1.0;  // Cannot estimate
    
    // Convert km/h to m/s
    double speedMPS = averageSpeed / 3.6;
    return distance / speedMPS;
}

CongestionLevel CongestionAnalyzer::predictCongestion(int futureFrames) {
    if (history.congestionHistory.size() < 10) {
        return CongestionLevel::FREE_FLOW;  // Not enough data
    }
    
    // Simple trend-based prediction
    double trend = calculateTrend(history.densityHistory, 10);
    
    double currentDensity = history.densityHistory.back();
    double predictedDensity = currentDensity + (trend * futureFrames);
    
    // Map predicted density to congestion level
    if (predictedDensity < 20) return CongestionLevel::FREE_FLOW;
    else if (predictedDensity < 40) return CongestionLevel::LIGHT;
    else if (predictedDensity < 60) return CongestionLevel::MODERATE;
    else if (predictedDensity < 80) return CongestionLevel::HEAVY;
    else return CongestionLevel::SEVERE;
}

bool CongestionAnalyzer::isCongestionIncreasing() {
    if (history.densityHistory.size() < 5) return false;
    
    double trend = calculateTrend(history.densityHistory, 5);
    return trend > 0.5;  // Positive trend indicates increasing congestion
}

bool CongestionAnalyzer::isCongestionDecreasing() {
    if (history.densityHistory.size() < 5) return false;
    
    double trend = calculateTrend(history.densityHistory, 5);
    return trend < -0.5;  // Negative trend indicates decreasing congestion
}

void CongestionAnalyzer::updateHistory(const CongestionMetrics& metrics) {
    addToHistory(history.densityHistory, metrics.densityScore, history.maxHistorySize);
    addToHistory(history.speedHistory, metrics.averageSpeed, history.maxHistorySize);
    
    history.vehicleCountHistory.push_back(metrics.vehicleCount);
    if (history.vehicleCountHistory.size() > static_cast<size_t>(history.maxHistorySize)) {
        history.vehicleCountHistory.erase(history.vehicleCountHistory.begin());
    }
    
    history.congestionHistory.push_back(metrics.congestionLevel);
    if (history.congestionHistory.size() > static_cast<size_t>(history.maxHistorySize)) {
        history.congestionHistory.erase(history.congestionHistory.begin());
    }
}

void CongestionAnalyzer::clearHistory() {
    history.densityHistory.clear();
    history.speedHistory.clear();
    history.vehicleCountHistory.clear();
    history.congestionHistory.clear();
}

std::string CongestionAnalyzer::generateRecommendation(const CongestionMetrics& metrics) {
    std::string recommendation;
    
    switch (metrics.congestionLevel) {
        case CongestionLevel::FREE_FLOW:
            recommendation = "Traffic flowing smoothly. No action needed.";
            break;
        case CongestionLevel::LIGHT:
            recommendation = "Light traffic. Monitor for changes.";
            break;
        case CongestionLevel::MODERATE:
            recommendation = "Moderate congestion. Consider alternate routes.";
            if (shouldAdjustSignalTiming(metrics)) {
                recommendation += " Adjust signal timing.";
            }
            break;
        case CongestionLevel::HEAVY:
            recommendation = "Heavy congestion. Recommend alternate routes.";
            if (shouldRerouteTraffic(metrics)) {
                recommendation += " Activate rerouting protocols.";
            }
            break;
        case CongestionLevel::SEVERE:
            recommendation = "SEVERE congestion! Immediate action required.";
            recommendation += " Activate emergency traffic management.";
            break;
    }
    
    if (isCongestionIncreasing()) {
        recommendation += " [TREND: Increasing]";
    } else if (isCongestionDecreasing()) {
        recommendation += " [TREND: Decreasing]";
    }
    
    return recommendation;
}

bool CongestionAnalyzer::shouldRerouteTraffic(const CongestionMetrics& metrics) {
    return metrics.congestionLevel >= CongestionLevel::HEAVY || 
           (metrics.occupancyRatio > 0.7 && isCongestionIncreasing());
}

bool CongestionAnalyzer::shouldAdjustSignalTiming(const CongestionMetrics& metrics) {
    return metrics.congestionLevel >= CongestionLevel::MODERATE &&
           metrics.vehicleCount > 20;
}

cv::Mat CongestionAnalyzer::visualizeCongestion(const cv::Mat& frame, const CongestionMetrics& metrics) {
    cv::Mat result = frame.clone();
    
    // Draw congestion level banner
    cv::Scalar bannerColor = getCongestionColor(metrics.congestionLevel);
    cv::rectangle(result, cv::Point(0, 0), cv::Point(result.cols, 100), bannerColor, -1);
    
    // Draw congestion level text
    std::string levelText = "Congestion: ";
    switch (metrics.congestionLevel) {
        case CongestionLevel::FREE_FLOW: levelText += "FREE FLOW"; break;
        case CongestionLevel::LIGHT: levelText += "LIGHT"; break;
        case CongestionLevel::MODERATE: levelText += "MODERATE"; break;
        case CongestionLevel::HEAVY: levelText += "HEAVY"; break;
        case CongestionLevel::SEVERE: levelText += "SEVERE"; break;
    }
    
    cv::putText(result, levelText, cv::Point(20, 40),
               cv::FONT_HERSHEY_BOLD, 1.2, cv::Scalar(255, 255, 255), 3);
    
    // Draw metrics
    int y = 70;
    std::string metricsText = "Vehicles: " + std::to_string(metrics.vehicleCount) +
                             " | Speed: " + std::to_string(static_cast<int>(metrics.averageSpeed)) + " km/h" +
                             " | Occupancy: " + std::to_string(static_cast<int>(metrics.occupancyRatio * 100)) + "%";
    cv::putText(result, metricsText, cv::Point(20, y),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    // Draw recommendation
    cv::rectangle(result, cv::Point(0, result.rows - 80), 
                 cv::Point(result.cols, result.rows), cv::Scalar(0, 0, 0), -1);
    
    // Word wrap recommendation
    std::string rec = metrics.recommendation;
    cv::putText(result, rec, cv::Point(20, result.rows - 50),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    return result;
}

cv::Mat CongestionAnalyzer::drawCongestionHeatmap(const cv::Mat& frame, const std::vector<Detection>& detections) {
    cv::Mat heatmap = cv::Mat::zeros(frame.size(), CV_8UC3);
    
    // Create density map
    for (const auto& detection : detections) {
        cv::circle(heatmap, 
                  cv::Point(detection.boundingBox.x + detection.boundingBox.width/2,
                           detection.boundingBox.y + detection.boundingBox.height/2),
                  50, cv::Scalar(0, 0, 255), -1);
    }
    
    // Blur to create heat effect
    cv::GaussianBlur(heatmap, heatmap, cv::Size(51, 51), 0);
    
    // Apply colormap
    cv::Mat heatmapGray;
    cv::cvtColor(heatmap, heatmapGray, cv::COLOR_BGR2GRAY);
    cv::applyColorMap(heatmapGray, heatmap, cv::COLORMAP_JET);
    
    // Blend with original frame
    cv::Mat result;
    cv::addWeighted(frame, 0.6, heatmap, 0.4, 0, result);
    
    return result;
}

cv::Mat CongestionAnalyzer::drawHistoricalTrends(int width, int height) {
    cv::Mat graph = cv::Mat::zeros(height, width, CV_8UC3);
    graph.setTo(cv::Scalar(255, 255, 255));
    
    if (history.densityHistory.empty()) {
        cv::putText(graph, "No historical data available", 
                   cv::Point(width/2 - 150, height/2),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        return graph;
    }
    
    // Draw axes
    cv::line(graph, cv::Point(50, height - 50), cv::Point(width - 50, height - 50), 
            cv::Scalar(0, 0, 0), 2);
    cv::line(graph, cv::Point(50, 50), cv::Point(50, height - 50), 
            cv::Scalar(0, 0, 0), 2);
    
    // Draw density trend
    if (history.densityHistory.size() > 1) {
        double maxDensity = *std::max_element(history.densityHistory.begin(), history.densityHistory.end());
        if (maxDensity == 0) maxDensity = 1;
        
        for (size_t i = 1; i < history.densityHistory.size(); ++i) {
            int x1 = 50 + ((i - 1) * (width - 100)) / history.densityHistory.size();
            int x2 = 50 + (i * (width - 100)) / history.densityHistory.size();
            int y1 = height - 50 - (history.densityHistory[i-1] / maxDensity) * (height - 100);
            int y2 = height - 50 - (history.densityHistory[i] / maxDensity) * (height - 100);
            
            cv::line(graph, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
        }
    }
    
    // Draw labels
    cv::putText(graph, "Density Over Time", cv::Point(width/2 - 100, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(graph, "Time", cv::Point(width/2 - 30, height - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    cv::putText(graph, "Density", cv::Point(5, height/2),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    
    return graph;
}

void CongestionAnalyzer::setRoadLength(double lengthInMeters) {
    this->roadLength = lengthInMeters;
}

void CongestionAnalyzer::setHistorySize(int maxSize) {
    history.maxHistorySize = maxSize;
}

void CongestionAnalyzer::setCongestionThresholds(double light, double moderate, double heavy) {
    lightCongestionThreshold = light;
    moderateCongestionThreshold = moderate;
    heavyCongestionThreshold = heavy;
}

std::map<CongestionLevel, int> CongestionAnalyzer::getCongestionStatistics() {
    std::map<CongestionLevel, int> stats;
    
    for (CongestionLevel level : history.congestionHistory) {
        stats[level]++;
    }
    
    return stats;
}

double CongestionAnalyzer::getAverageCongestionLevel() {
    if (history.densityHistory.empty()) return 0.0;
    
    double sum = std::accumulate(history.densityHistory.begin(), history.densityHistory.end(), 0.0);
    return sum / history.densityHistory.size();
}

cv::Scalar CongestionAnalyzer::getCongestionColor(CongestionLevel level) {
    switch (level) {
        case CongestionLevel::FREE_FLOW:
            return cv::Scalar(0, 255, 0);  // Green
        case CongestionLevel::LIGHT:
            return cv::Scalar(0, 255, 255);  // Yellow
        case CongestionLevel::MODERATE:
            return cv::Scalar(0, 165, 255);  // Orange
        case CongestionLevel::HEAVY:
            return cv::Scalar(0, 69, 255);  // Dark Orange
        case CongestionLevel::SEVERE:
            return cv::Scalar(0, 0, 255);  // Red
        default:
            return cv::Scalar(128, 128, 128);  // Gray
    }
}

cv::Scalar CongestionAnalyzer::getTrafficStateColor(TrafficState state) {
    switch (state) {
        case TrafficState::FREE_FLOW:
            return cv::Scalar(0, 255, 0);
        case TrafficState::SYNCHRONIZED:
            return cv::Scalar(0, 255, 255);
        case TrafficState::WIDE_MOVING_JAM:
            return cv::Scalar(0, 165, 255);
        case TrafficState::STOPPED:
            return cv::Scalar(0, 0, 255);
        default:
            return cv::Scalar(128, 128, 128);
    }
}

double CongestionAnalyzer::calculateTrend(const std::vector<double>& data, int windowSize) {
    if (data.size() < static_cast<size_t>(windowSize)) {
        return 0.0;
    }
    
    // Simple linear regression on last windowSize points
    int n = windowSize;
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    for (int i = 0; i < n; ++i) {
        int idx = data.size() - n + i;
        double x = i;
        double y = data[idx];
        
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumX2 += x * x;
    }
    
    double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return slope;
}

void CongestionAnalyzer::addToHistory(std::vector<double>& history, double value, int maxSize) {
    history.push_back(value);
    if (history.size() > static_cast<size_t>(maxSize)) {
        history.erase(history.begin());
    }
}

} // namespace traffic
