#include "visualization.h"
#include <algorithm>
#include <cmath>

namespace traffic {

// Define static color palettes
const std::vector<cv::Scalar> Visualization::CLASS_COLORS = {
    cv::Scalar(255, 0, 0),      // Blue
    cv::Scalar(0, 255, 0),      // Green
    cv::Scalar(0, 0, 255),      // Red
    cv::Scalar(255, 255, 0),    // Cyan
    cv::Scalar(255, 0, 255),    // Magenta
    cv::Scalar(0, 255, 255),    // Yellow
    cv::Scalar(128, 0, 128),    // Purple
    cv::Scalar(255, 165, 0)     // Orange
};

const std::map<CongestionLevel, cv::Scalar> Visualization::CONGESTION_COLORS = {
    {CongestionLevel::FREE_FLOW, cv::Scalar(0, 255, 0)},      // Green
    {CongestionLevel::LIGHT, cv::Scalar(0, 255, 255)},        // Yellow
    {CongestionLevel::MODERATE, cv::Scalar(0, 165, 255)},     // Orange
    {CongestionLevel::HEAVY, cv::Scalar(0, 69, 255)},         // Dark Orange
    {CongestionLevel::SEVERE, cv::Scalar(0, 0, 255)}          // Red
};

void Visualization::drawBoundingBox(cv::Mat& image, const cv::Rect& box, 
                                   const cv::Scalar& color, int thickness) {
    cv::rectangle(image, box, color, thickness);
}

void Visualization::drawDetections(cv::Mat& image, const std::vector<Detection>& detections,
                                  bool showLabels, bool showConfidence) {
    for (const auto& detection : detections) {
        cv::Scalar color = getColorForClass(detection.classId);
        
        // Draw bounding box
        drawBoundingBox(image, detection.boundingBox, color, 2);
        
        // Draw label
        if (showLabels) {
            std::string label = detection.className;
            if (showConfidence) {
                label += " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
            }
            
            cv::Point labelPos(detection.boundingBox.x, detection.boundingBox.y - 5);
            drawTextWithBackground(image, label, labelPos, cv::Scalar(255, 255, 255), color);
        }
    }
}

void Visualization::drawLabel(cv::Mat& image, const std::string& label, 
                             const cv::Point& position, 
                             const cv::Scalar& bgColor,
                             const cv::Scalar& textColor) {
    drawTextWithBackground(image, label, position, textColor, bgColor);
}

cv::Mat Visualization::createDensityHeatmap(const cv::Mat& image, 
                                           const std::vector<Detection>& detections,
                                           int blurSize) {
    cv::Mat heatmap = cv::Mat::zeros(image.size(), CV_8UC1);
    
    // Draw circles at detection centers
    for (const auto& detection : detections) {
        cv::Point center(detection.boundingBox.x + detection.boundingBox.width / 2,
                        detection.boundingBox.y + detection.boundingBox.height / 2);
        cv::circle(heatmap, center, 50, cv::Scalar(255), -1);
    }
    
    // Apply Gaussian blur
    if (blurSize > 0) {
        cv::GaussianBlur(heatmap, heatmap, cv::Size(blurSize, blurSize), 0);
    }
    
    // Apply colormap
    cv::Mat coloredHeatmap;
    cv::applyColorMap(heatmap, coloredHeatmap, cv::COLORMAP_JET);
    
    // Blend with original image
    cv::Mat result;
    cv::addWeighted(image, 0.6, coloredHeatmap, 0.4, 0, result);
    
    return result;
}

cv::Mat Visualization::drawDensityGrid(const cv::Mat& image, 
                                      const std::vector<std::vector<int>>& gridDensity,
                                      int gridRows, int gridCols) {
    cv::Mat result = image.clone();
    
    int cellHeight = image.rows / gridRows;
    int cellWidth = image.cols / gridCols;
    
    // Find max density for normalization
    int maxDensity = 0;
    for (const auto& row : gridDensity) {
        for (int density : row) {
            maxDensity = std::max(maxDensity, density);
        }
    }
    
    if (maxDensity == 0) maxDensity = 1;
    
    // Draw grid cells
    for (int i = 0; i < gridRows; ++i) {
        for (int j = 0; j < gridCols; ++j) {
            int density = gridDensity[i][j];
            double ratio = static_cast<double>(density) / maxDensity;
            
            cv::Rect cell(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
            cv::Scalar color = interpolateColor(cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), ratio);
            
            drawSemiTransparentRect(result, cell, color, 0.3);
            
            // Draw density number
            std::string densityText = std::to_string(density);
            cv::Point textPos(cell.x + cellWidth / 2 - 10, cell.y + cellHeight / 2);
            cv::putText(result, densityText, textPos, cv::FONT_HERSHEY_SIMPLEX, 
                       0.5, cv::Scalar(255, 255, 255), 2);
        }
    }
    
    return result;
}

void Visualization::drawMetricsPanel(cv::Mat& image, 
                                    const std::map<std::string, std::string>& metrics,
                                    const cv::Point& position,
                                    const cv::Scalar& bgColor) {
    int y = position.y;
    int lineHeight = 30;
    int maxWidth = 0;
    
    // Calculate panel size
    for (const auto& metric : metrics) {
        std::string text = metric.first + ": " + metric.second;
        cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2);
        maxWidth = std::max(maxWidth, textSize.width);
    }
    
    // Draw background
    cv::Rect panelRect(position.x - 10, position.y - 20, 
                      maxWidth + 20, metrics.size() * lineHeight + 20);
    drawSemiTransparentRect(image, panelRect, bgColor, 0.7);
    
    // Draw metrics
    for (const auto& metric : metrics) {
        std::string text = metric.first + ": " + metric.second;
        cv::putText(image, text, cv::Point(position.x, y), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        y += lineHeight;
    }
}

void Visualization::drawProgressBar(cv::Mat& image, const std::string& label,
                                   double value, double maxValue,
                                   const cv::Point& position,
                                   const cv::Size& size) {
    // Draw label
    cv::putText(image, label, position, cv::FONT_HERSHEY_SIMPLEX, 
               0.5, cv::Scalar(255, 255, 255), 1);
    
    // Draw bar background
    cv::Point barPos(position.x, position.y + 20);
    cv::rectangle(image, barPos, cv::Point(barPos.x + size.width, barPos.y + size.height),
                 cv::Scalar(100, 100, 100), -1);
    
    // Draw filled portion
    double ratio = std::min(1.0, value / maxValue);
    int fillWidth = static_cast<int>(size.width * ratio);
    cv::Scalar fillColor = interpolateColor(cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), ratio);
    cv::rectangle(image, barPos, cv::Point(barPos.x + fillWidth, barPos.y + size.height),
                 fillColor, -1);
    
    // Draw border
    cv::rectangle(image, barPos, cv::Point(barPos.x + size.width, barPos.y + size.height),
                 cv::Scalar(255, 255, 255), 2);
    
    // Draw value text
    std::string valueText = std::to_string(static_cast<int>(value)) + "/" + 
                           std::to_string(static_cast<int>(maxValue));
    cv::putText(image, valueText, cv::Point(barPos.x + size.width + 10, barPos.y + size.height - 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

cv::Mat Visualization::createLineChart(const std::vector<double>& data,
                                      const std::string& title,
                                      const cv::Size& size,
                                      const cv::Scalar& lineColor) {
    cv::Mat chart = cv::Mat::zeros(size, CV_8UC3);
    chart.setTo(cv::Scalar(255, 255, 255));
    
    if (data.empty()) {
        cv::putText(chart, "No data available", cv::Point(size.width/2 - 100, size.height/2),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
        return chart;
    }
    
    // Draw title
    cv::putText(chart, title, cv::Point(size.width/2 - 100, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Draw axes
    int margin = 50;
    cv::line(chart, cv::Point(margin, size.height - margin), 
            cv::Point(size.width - margin, size.height - margin), 
            cv::Scalar(0, 0, 0), 2);
    cv::line(chart, cv::Point(margin, margin), 
            cv::Point(margin, size.height - margin), 
            cv::Scalar(0, 0, 0), 2);
    
    // Find min and max values
    double minVal = *std::min_element(data.begin(), data.end());
    double maxVal = *std::max_element(data.begin(), data.end());
    if (maxVal == minVal) maxVal = minVal + 1;
    
    // Draw data points
    int chartWidth = size.width - 2 * margin;
    int chartHeight = size.height - 2 * margin;
    
    for (size_t i = 1; i < data.size(); ++i) {
        int x1 = margin + ((i - 1) * chartWidth) / (data.size() - 1);
        int x2 = margin + (i * chartWidth) / (data.size() - 1);
        int y1 = size.height - margin - ((data[i-1] - minVal) / (maxVal - minVal)) * chartHeight;
        int y2 = size.height - margin - ((data[i] - minVal) / (maxVal - minVal)) * chartHeight;
        
        cv::line(chart, cv::Point(x1, y1), cv::Point(x2, y2), lineColor, 2);
    }
    
    return chart;
}

cv::Mat Visualization::createBarChart(const std::map<std::string, int>& data,
                                     const std::string& title,
                                     const cv::Size& size) {
    cv::Mat chart = cv::Mat::zeros(size, CV_8UC3);
    chart.setTo(cv::Scalar(255, 255, 255));
    
    if (data.empty()) {
        return chart;
    }
    
    // Draw title
    cv::putText(chart, title, cv::Point(size.width/2 - 100, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Find max value
    int maxVal = 0;
    for (const auto& pair : data) {
        maxVal = std::max(maxVal, pair.second);
    }
    if (maxVal == 0) maxVal = 1;
    
    // Draw bars
    int margin = 50;
    int barWidth = (size.width - 2 * margin) / data.size();
    int chartHeight = size.height - 2 * margin;
    int x = margin;
    
    int colorIdx = 0;
    for (const auto& pair : data) {
        int barHeight = (pair.second * chartHeight) / maxVal;
        cv::Rect bar(x, size.height - margin - barHeight, barWidth - 10, barHeight);
        
        cv::Scalar color = CLASS_COLORS[colorIdx % CLASS_COLORS.size()];
        cv::rectangle(chart, bar, color, -1);
        
        // Draw label
        cv::putText(chart, pair.first, cv::Point(x, size.height - margin + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        
        // Draw value
        cv::putText(chart, std::to_string(pair.second), 
                   cv::Point(x + barWidth/2 - 10, bar.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        x += barWidth;
        colorIdx++;
    }
    
    return chart;
}

cv::Mat Visualization::createPieChart(const std::map<std::string, int>& data,
                                     const std::string& title,
                                     const cv::Size& size) {
    cv::Mat chart = cv::Mat::zeros(size, CV_8UC3);
    chart.setTo(cv::Scalar(255, 255, 255));
    
    if (data.empty()) {
        return chart;
    }
    
    // Calculate total
    int total = 0;
    for (const auto& pair : data) {
        total += pair.second;
    }
    if (total == 0) total = 1;
    
    // Draw title
    cv::putText(chart, title, cv::Point(size.width/2 - 100, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Draw pie slices
    cv::Point center(size.width / 2, size.height / 2);
    int radius = std::min(size.width, size.height) / 3;
    
    double startAngle = 0;
    int colorIdx = 0;
    
    for (const auto& pair : data) {
        double angle = (pair.second * 360.0) / total;
        cv::Scalar color = CLASS_COLORS[colorIdx % CLASS_COLORS.size()];
        
        cv::ellipse(chart, center, cv::Size(radius, radius), 0, 
                   startAngle, startAngle + angle, color, -1);
        
        startAngle += angle;
        colorIdx++;
    }
    
    // Draw legend
    int legendY = 60;
    colorIdx = 0;
    for (const auto& pair : data) {
        cv::Scalar color = CLASS_COLORS[colorIdx % CLASS_COLORS.size()];
        cv::rectangle(chart, cv::Point(10, legendY), cv::Point(30, legendY + 15), color, -1);
        
        std::string label = pair.first + ": " + std::to_string(pair.second);
        cv::putText(chart, label, cv::Point(35, legendY + 12),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        
        legendY += 20;
        colorIdx++;
    }
    
    return chart;
}

void Visualization::drawStatusIndicator(cv::Mat& image, const std::string& status,
                                       const cv::Point& position,
                                       const cv::Scalar& color) {
    // Draw circle indicator
    cv::circle(image, position, 10, color, -1);
    cv::circle(image, position, 10, cv::Scalar(255, 255, 255), 2);
    
    // Draw status text
    cv::putText(image, status, cv::Point(position.x + 20, position.y + 5),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}

void Visualization::drawCongestionBanner(cv::Mat& image, CongestionLevel level,
                                        const std::string& additionalInfo) {
    cv::Scalar color = getColorForCongestion(level);
    
    // Draw banner background
    cv::rectangle(image, cv::Point(0, 0), cv::Point(image.cols, 80), color, -1);
    
    // Draw congestion level text
    std::string levelText;
    switch (level) {
        case CongestionLevel::FREE_FLOW: levelText = "FREE FLOW"; break;
        case CongestionLevel::LIGHT: levelText = "LIGHT CONGESTION"; break;
        case CongestionLevel::MODERATE: levelText = "MODERATE CONGESTION"; break;
        case CongestionLevel::HEAVY: levelText = "HEAVY CONGESTION"; break;
        case CongestionLevel::SEVERE: levelText = "SEVERE CONGESTION"; break;
    }
    
    cv::putText(image, levelText, cv::Point(20, 40),
               cv::FONT_HERSHEY_BOLD, 1.2, cv::Scalar(255, 255, 255), 3);
    
    if (!additionalInfo.empty()) {
        cv::putText(image, additionalInfo, cv::Point(20, 70),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

void Visualization::drawROI(cv::Mat& image, const cv::Rect& roi,
                           const cv::Scalar& color, int thickness) {
    cv::rectangle(image, roi, color, thickness);
    cv::putText(image, "ROI", cv::Point(roi.x + 5, roi.y + 20),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}

void Visualization::drawMultipleROIs(cv::Mat& image, 
                                    const std::vector<cv::Rect>& rois,
                                    const std::vector<std::string>& labels) {
    for (size_t i = 0; i < rois.size(); ++i) {
        cv::Scalar color = CLASS_COLORS[i % CLASS_COLORS.size()];
        cv::rectangle(image, rois[i], color, 2);
        
        std::string label = (i < labels.size()) ? labels[i] : "ROI " + std::to_string(i + 1);
        cv::putText(image, label, cv::Point(rois[i].x + 5, rois[i].y + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

cv::Mat Visualization::createOverlay(const cv::Mat& background, 
                                    const cv::Mat& overlay,
                                    double alpha) {
    cv::Mat result;
    cv::addWeighted(background, 1.0 - alpha, overlay, alpha, 0, result);
    return result;
}

void Visualization::drawSemiTransparentRect(cv::Mat& image, const cv::Rect& rect,
                                           const cv::Scalar& color, double alpha) {
    cv::Mat overlay = image.clone();
    cv::rectangle(overlay, rect, color, -1);
    cv::addWeighted(image, 1.0 - alpha, overlay, alpha, 0, image);
}

cv::Scalar Visualization::getColorForClass(int classId) {
    return CLASS_COLORS[classId % CLASS_COLORS.size()];
}

cv::Scalar Visualization::getColorForCongestion(CongestionLevel level) {
    auto it = CONGESTION_COLORS.find(level);
    if (it != CONGESTION_COLORS.end()) {
        return it->second;
    }
    return cv::Scalar(128, 128, 128);  // Gray for unknown
}

cv::Scalar Visualization::interpolateColor(const cv::Scalar& color1, 
                                          const cv::Scalar& color2, 
                                          double ratio) {
    ratio = std::max(0.0, std::min(1.0, ratio));
    return cv::Scalar(
        color1[0] + (color2[0] - color1[0]) * ratio,
        color1[1] + (color2[1] - color1[1]) * ratio,
        color1[2] + (color2[2] - color1[2]) * ratio
    );
}

cv::Size Visualization::getTextSize(const std::string& text, 
                                   int fontFace, double fontScale, int thickness) {
    int baseline;
    return cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
}

void Visualization::drawTextWithBackground(cv::Mat& image, const std::string& text,
                                          const cv::Point& position,
                                          const cv::Scalar& textColor,
                                          const cv::Scalar& bgColor,
                                          double fontScale, int thickness) {
    cv::Size textSize = getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness);
    
    cv::Rect bgRect(position.x - 2, position.y - textSize.height - 2,
                   textSize.width + 4, textSize.height + 4);
    cv::rectangle(image, bgRect, bgColor, -1);
    
    cv::putText(image, text, position, cv::FONT_HERSHEY_SIMPLEX, 
               fontScale, textColor, thickness);
}

cv::Mat Visualization::createDashboard(const cv::Mat& mainView,
                                      const std::vector<cv::Mat>& additionalViews,
                                      const std::map<std::string, std::string>& metrics) {
    // Create dashboard layout
    int dashboardWidth = mainView.cols + 400;
    int dashboardHeight = std::max(mainView.rows, 800);
    
    cv::Mat dashboard = cv::Mat::zeros(dashboardHeight, dashboardWidth, CV_8UC3);
    dashboard.setTo(cv::Scalar(50, 50, 50));
    
    // Place main view
    mainView.copyTo(dashboard(cv::Rect(0, 0, mainView.cols, mainView.rows)));
    
    // Place metrics panel
    drawMetricsPanel(dashboard, metrics, cv::Point(mainView.cols + 20, 30));
    
    // Place additional views
    int y = 300;
    for (const auto& view : additionalViews) {
        if (y + view.rows < dashboardHeight) {
            view.copyTo(dashboard(cv::Rect(mainView.cols + 20, y, view.cols, view.rows)));
            y += view.rows + 20;
        }
    }
    
    return dashboard;
}

void Visualization::addTimestamp(cv::Mat& image, const std::string& timestamp,
                                const cv::Point& position) {
    cv::Point pos = position;
    if (pos.x < 0 || pos.y < 0) {
        pos = cv::Point(image.cols - 200, image.rows - 20);
    }
    
    drawTextWithBackground(image, timestamp, pos, cv::Scalar(255, 255, 255), 
                          cv::Scalar(0, 0, 0), 0.5, 1);
}

void Visualization::addWatermark(cv::Mat& image, const std::string& watermark,
                                const cv::Point& position) {
    cv::Point pos = position;
    if (pos.x < 0 || pos.y < 0) {
        pos = cv::Point(image.cols - 300, 30);
    }
    
    cv::putText(image, watermark, pos, cv::FONT_HERSHEY_SIMPLEX, 
               0.7, cv::Scalar(255, 255, 255), 2);
}

} // namespace traffic
