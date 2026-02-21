#include "speed_tracker.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace traffic {

SpeedTracker::SpeedTracker() 
    : nextVehicleId(0), pixelsPerMeter(10.0), fps(30.0), 
      minSpeedThreshold(5.0), maxSpeedThreshold(200.0),
      maxDistanceThreshold(50.0), maxInactiveFrames(30) {
}

SpeedTracker::SpeedTracker(double pixelsPerMeter, double fps)
    : nextVehicleId(0), pixelsPerMeter(pixelsPerMeter), fps(fps),
      minSpeedThreshold(5.0), maxSpeedThreshold(200.0),
      maxDistanceThreshold(50.0), maxInactiveFrames(30) {
}

void SpeedTracker::update(const cv::Mat& frame, const std::vector<Detection>& detections) {
    // Mark all vehicles as inactive initially
    for (auto& pair : vehicles) {
        pair.second.active = false;
    }
    
    // Match detections to existing vehicles
    std::vector<bool> matched(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        int vehicleId = matchDetectionToVehicle(detections[i]);
        
        if (vehicleId >= 0) {
            // Update existing vehicle
            TrackedVehicle& vehicle = vehicles[vehicleId];
            vehicle.boundingBox = detections[i].boundingBox;
            vehicle.trajectory.push_back(getCenter(detections[i].boundingBox));
            vehicle.frameCount++;
            vehicle.active = true;
            
            // Keep trajectory size manageable
            if (vehicle.trajectory.size() > 30) {
                vehicle.trajectory.erase(vehicle.trajectory.begin());
            }
            
            updateVehicleSpeed(vehicle);
            matched[i] = true;
        }
    }
    
    // Create new vehicles for unmatched detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matched[i]) {
            TrackedVehicle newVehicle;
            newVehicle.id = nextVehicleId++;
            newVehicle.boundingBox = detections[i].boundingBox;
            newVehicle.trajectory.push_back(getCenter(detections[i].boundingBox));
            newVehicle.speed = 0.0;
            newVehicle.averageSpeed = 0.0;
            newVehicle.frameCount = 1;
            newVehicle.active = true;
            newVehicle.classId = detections[i].classId;
            newVehicle.className = detections[i].className;
            
            vehicles[newVehicle.id] = newVehicle;
        }
    }
    
    // Store current frame for next iteration
    frame.copyTo(prevFrame);
    
    // Remove inactive vehicles
    removeInactiveVehicles(maxInactiveFrames);
}

std::vector<TrackedVehicle> SpeedTracker::getTrackedVehicles() const {
    std::vector<TrackedVehicle> result;
    for (const auto& pair : vehicles) {
        if (pair.second.active) {
            result.push_back(pair.second);
        }
    }
    return result;
}

SpeedMetrics SpeedTracker::getSpeedMetrics() const {
    SpeedMetrics metrics;
    metrics.averageSpeed = 0.0;
    metrics.maxSpeed = 0.0;
    metrics.minSpeed = 1000.0;
    metrics.totalVehicles = vehicles.size();
    metrics.activeVehicles = 0;
    
    std::map<int, std::vector<double>> speedsByClass;
    
    for (const auto& pair : vehicles) {
        const TrackedVehicle& vehicle = pair.second;
        
        if (vehicle.active && vehicle.speed > 0) {
            metrics.activeVehicles++;
            metrics.averageSpeed += vehicle.speed;
            metrics.maxSpeed = std::max(metrics.maxSpeed, vehicle.speed);
            metrics.minSpeed = std::min(metrics.minSpeed, vehicle.speed);
            
            speedsByClass[vehicle.classId].push_back(vehicle.speed);
        }
    }
    
    if (metrics.activeVehicles > 0) {
        metrics.averageSpeed /= metrics.activeVehicles;
    }
    
    // Calculate average speed by class
    for (const auto& pair : speedsByClass) {
        double sum = 0.0;
        for (double speed : pair.second) {
            sum += speed;
        }
        metrics.speedByClass[pair.first] = sum / pair.second.size();
    }
    
    return metrics;
}

void SpeedTracker::calibratePixelsPerMeter(double pixelsPerMeter) {
    this->pixelsPerMeter = pixelsPerMeter;
}

void SpeedTracker::setFPS(double fps) {
    this->fps = fps;
}

void SpeedTracker::setSpeedLimits(double minSpeed, double maxSpeed) {
    this->minSpeedThreshold = minSpeed;
    this->maxSpeedThreshold = maxSpeed;
}

double SpeedTracker::calculateSpeed(const TrackedVehicle& vehicle) {
    if (vehicle.trajectory.size() < 2) {
        return 0.0;
    }
    
    // Calculate speed based on last few positions
    int numPoints = std::min(5, static_cast<int>(vehicle.trajectory.size()));
    cv::Point2f start = vehicle.trajectory[vehicle.trajectory.size() - numPoints];
    cv::Point2f end = vehicle.trajectory.back();
    
    double pixelDistance = calculateDistance(start, end);
    double meters = pixelDistance / pixelsPerMeter;
    double seconds = numPoints / fps;
    
    if (seconds > 0) {
        double metersPerSecond = meters / seconds;
        double kmPerHour = metersPerSecond * 3.6;
        
        // Apply speed limits
        kmPerHour = std::max(minSpeedThreshold, std::min(maxSpeedThreshold, kmPerHour));
        return kmPerHour;
    }
    
    return 0.0;
}

double SpeedTracker::estimateSpeedFromOpticalFlow(const cv::Mat& prevFrame, const cv::Mat& currFrame, const cv::Rect& roi) {
    if (prevFrame.empty() || currFrame.empty()) {
        return 0.0;
    }
    
    cv::Mat prevGray, currGray;
    cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);
    
    // Extract ROI
    cv::Mat prevROI = prevGray(roi);
    cv::Mat currROI = currGray(roi);
    
    // Calculate optical flow
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prevROI, currROI, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    // Calculate average flow magnitude
    double totalMagnitude = 0.0;
    int count = 0;
    
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            cv::Point2f flowAtPoint = flow.at<cv::Point2f>(y, x);
            double magnitude = std::sqrt(flowAtPoint.x * flowAtPoint.x + flowAtPoint.y * flowAtPoint.y);
            totalMagnitude += magnitude;
            count++;
        }
    }
    
    if (count > 0) {
        double avgPixelsPerFrame = totalMagnitude / count;
        double metersPerFrame = avgPixelsPerFrame / pixelsPerMeter;
        double metersPerSecond = metersPerFrame * fps;
        return metersPerSecond * 3.6;  // Convert to km/h
    }
    
    return 0.0;
}

cv::Mat SpeedTracker::drawTracks(const cv::Mat& frame) {
    cv::Mat result = frame.clone();
    
    for (const auto& pair : vehicles) {
        const TrackedVehicle& vehicle = pair.second;
        
        if (!vehicle.active || vehicle.trajectory.size() < 2) {
            continue;
        }
        
        // Draw trajectory
        for (size_t i = 1; i < vehicle.trajectory.size(); ++i) {
            cv::line(result, vehicle.trajectory[i-1], vehicle.trajectory[i], 
                    cv::Scalar(0, 255, 255), 2);
        }
        
        // Draw bounding box
        cv::rectangle(result, vehicle.boundingBox, cv::Scalar(0, 255, 0), 2);
        
        // Draw vehicle ID and speed
        std::string label = "ID:" + std::to_string(vehicle.id) + 
                          " " + std::to_string(static_cast<int>(vehicle.speed)) + " km/h";
        cv::putText(result, label, 
                   cv::Point(vehicle.boundingBox.x, vehicle.boundingBox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    
    return result;
}

cv::Mat SpeedTracker::drawSpeedInfo(const cv::Mat& frame) {
    cv::Mat result = drawTracks(frame);
    
    SpeedMetrics metrics = getSpeedMetrics();
    
    // Draw metrics on frame
    int y = 30;
    cv::putText(result, "Active Vehicles: " + std::to_string(metrics.activeVehicles),
               cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    y += 30;
    
    cv::putText(result, "Avg Speed: " + std::to_string(static_cast<int>(metrics.averageSpeed)) + " km/h",
               cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    y += 30;
    
    cv::putText(result, "Max Speed: " + std::to_string(static_cast<int>(metrics.maxSpeed)) + " km/h",
               cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return result;
}

std::vector<TrackedVehicle> SpeedTracker::detectSpeedViolations(double speedLimit) {
    std::vector<TrackedVehicle> violations;
    
    for (const auto& pair : vehicles) {
        if (pair.second.active && isOverSpeed(pair.second, speedLimit)) {
            violations.push_back(pair.second);
        }
    }
    
    return violations;
}

bool SpeedTracker::isOverSpeed(const TrackedVehicle& vehicle, double speedLimit) {
    return vehicle.speed > speedLimit;
}

void SpeedTracker::reset() {
    vehicles.clear();
    nextVehicleId = 0;
    prevFrame.release();
}

void SpeedTracker::removeInactiveVehicles(int maxInactiveFrames) {
    auto it = vehicles.begin();
    while (it != vehicles.end()) {
        if (!it->second.active) {
            it = vehicles.erase(it);
        } else {
            ++it;
        }
    }
}

int SpeedTracker::matchDetectionToVehicle(const Detection& detection) {
    int bestMatch = -1;
    double minDistance = maxDistanceThreshold;
    
    cv::Point2f detectionCenter = getCenter(detection.boundingBox);
    
    for (auto& pair : vehicles) {
        if (pair.second.trajectory.empty()) {
            continue;
        }
        
        cv::Point2f vehicleCenter = pair.second.trajectory.back();
        double distance = calculateDistance(detectionCenter, vehicleCenter);
        
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = pair.first;
        }
    }
    
    return bestMatch;
}

double SpeedTracker::calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f SpeedTracker::getCenter(const cv::Rect& rect) {
    return cv::Point2f(rect.x + rect.width / 2.0f, rect.y + rect.height / 2.0f);
}

void SpeedTracker::updateVehicleSpeed(TrackedVehicle& vehicle) {
    vehicle.speed = calculateSpeed(vehicle);
    
    // Update average speed
    if (vehicle.frameCount > 0) {
        vehicle.averageSpeed = ((vehicle.averageSpeed * (vehicle.frameCount - 1)) + vehicle.speed) / vehicle.frameCount;
    }
}

cv::Scalar SpeedTracker::getSpeedColor(double speed, double speedLimit) {
    if (speed > speedLimit) {
        return cv::Scalar(0, 0, 255);  // Red for over speed
    } else if (speed > speedLimit * 0.8) {
        return cv::Scalar(0, 165, 255);  // Orange for near limit
    } else {
        return cv::Scalar(0, 255, 0);  // Green for normal
    }
}

} // namespace traffic
