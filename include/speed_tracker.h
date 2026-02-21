#ifndef SPEED_TRACKER_H
#define SPEED_TRACKER_H

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <map>
#include <string>
#include "vehicle_detector.h"

namespace traffic {

struct TrackedVehicle {
    int id;
    cv::Rect boundingBox;
    std::vector<cv::Point2f> trajectory;
    double speed;  // in km/h
    double averageSpeed;
    int frameCount;
    bool active;
    int classId;
    std::string className;
};

struct SpeedMetrics {
    double averageSpeed;
    double maxSpeed;
    double minSpeed;
    int totalVehicles;
    int activeVehicles;
    std::map<int, double> speedByClass;  // Average speed per vehicle class
};

class SpeedTracker {
public:
    SpeedTracker();
    explicit SpeedTracker(double pixelsPerMeter, double fps = 30.0);
    
    // Tracking methods
    void update(const cv::Mat& frame, const std::vector<Detection>& detections);
    std::vector<TrackedVehicle> getTrackedVehicles() const;
    SpeedMetrics getSpeedMetrics() const;
    
    // Calibration
    void calibratePixelsPerMeter(double pixelsPerMeter);
    void setFPS(double fps);
    void setSpeedLimits(double minSpeed, double maxSpeed);
    
    // Speed calculation
    double calculateSpeed(const TrackedVehicle& vehicle);
    double estimateSpeedFromOpticalFlow(const cv::Mat& prevFrame, const cv::Mat& currFrame, const cv::Rect& roi);
    
    // Visualization
    cv::Mat drawTracks(const cv::Mat& frame);
    cv::Mat drawSpeedInfo(const cv::Mat& frame);
    
    // Violation detection
    std::vector<TrackedVehicle> detectSpeedViolations(double speedLimit);
    bool isOverSpeed(const TrackedVehicle& vehicle, double speedLimit);
    
    // Reset and cleanup
    void reset();
    void removeInactiveVehicles(int maxInactiveFrames = 30);
    
private:
    std::map<int, TrackedVehicle> vehicles;
    int nextVehicleId;
    double pixelsPerMeter;
    double fps;
    double minSpeedThreshold;
    double maxSpeedThreshold;
    
    cv::Mat prevFrame;
    cv::Ptr<cv::Tracker> tracker;
    
    // Tracking parameters
    double maxDistanceThreshold;  // Maximum distance for matching detections
    int maxInactiveFrames;
    
    // Helper methods
    int matchDetectionToVehicle(const Detection& detection);
    double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2);
    cv::Point2f getCenter(const cv::Rect& rect);
    void updateVehicleSpeed(TrackedVehicle& vehicle);
    cv::Scalar getSpeedColor(double speed, double speedLimit);
};

} // namespace traffic

#endif // SPEED_TRACKER_H
