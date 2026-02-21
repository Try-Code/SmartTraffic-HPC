#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "vehicle_detector.h"
#include "density_estimator.h"

namespace traffic {

class Visualization {
public:
    // Drawing utilities
    static void drawBoundingBox(cv::Mat& image, const cv::Rect& box, 
                               const cv::Scalar& color = cv::Scalar(0, 255, 0), 
                               int thickness = 2);
    
    static void drawDetections(cv::Mat& image, const std::vector<Detection>& detections,
                              bool showLabels = true, bool showConfidence = true);
    
    static void drawLabel(cv::Mat& image, const std::string& label, 
                         const cv::Point& position, 
                         const cv::Scalar& bgColor = cv::Scalar(0, 0, 0),
                         const cv::Scalar& textColor = cv::Scalar(255, 255, 255));
    
    // Density visualization
    static cv::Mat createDensityHeatmap(const cv::Mat& image, 
                                       const std::vector<Detection>& detections,
                                       int blurSize = 51);
    
    static cv::Mat drawDensityGrid(const cv::Mat& image, 
                                  const std::vector<std::vector<int>>& gridDensity,
                                  int gridRows, int gridCols);
    
    // Metrics visualization
    static void drawMetricsPanel(cv::Mat& image, 
                                const std::map<std::string, std::string>& metrics,
                                const cv::Point& position = cv::Point(10, 30),
                                const cv::Scalar& bgColor = cv::Scalar(0, 0, 0, 180));
    
    static void drawProgressBar(cv::Mat& image, const std::string& label,
                               double value, double maxValue,
                               const cv::Point& position,
                               const cv::Size& size = cv::Size(200, 20));
    
    // Charts and graphs
    static cv::Mat createLineChart(const std::vector<double>& data,
                                  const std::string& title,
                                  const cv::Size& size = cv::Size(800, 400),
                                  const cv::Scalar& lineColor = cv::Scalar(255, 0, 0));
    
    static cv::Mat createBarChart(const std::map<std::string, int>& data,
                                 const std::string& title,
                                 const cv::Size& size = cv::Size(800, 400));
    
    static cv::Mat createPieChart(const std::map<std::string, int>& data,
                                 const std::string& title,
                                 const cv::Size& size = cv::Size(400, 400));
    
    // Status indicators
    static void drawStatusIndicator(cv::Mat& image, const std::string& status,
                                   const cv::Point& position,
                                   const cv::Scalar& color);
    
    static void drawCongestionBanner(cv::Mat& image, CongestionLevel level,
                                    const std::string& additionalInfo = "");
    
    // ROI visualization
    static void drawROI(cv::Mat& image, const cv::Rect& roi,
                       const cv::Scalar& color = cv::Scalar(255, 255, 0),
                       int thickness = 2);
    
    static void drawMultipleROIs(cv::Mat& image, 
                                const std::vector<cv::Rect>& rois,
                                const std::vector<std::string>& labels = {});
    
    // Overlay utilities
    static cv::Mat createOverlay(const cv::Mat& background, 
                                const cv::Mat& overlay,
                                double alpha = 0.5);
    
    static void drawSemiTransparentRect(cv::Mat& image, const cv::Rect& rect,
                                       const cv::Scalar& color, double alpha = 0.5);
    
    // Color utilities
    static cv::Scalar getColorForClass(int classId);
    static cv::Scalar getColorForCongestion(CongestionLevel level);
    static cv::Scalar interpolateColor(const cv::Scalar& color1, 
                                      const cv::Scalar& color2, 
                                      double ratio);
    
    // Text utilities
    static cv::Size getTextSize(const std::string& text, 
                               int fontFace = cv::FONT_HERSHEY_SIMPLEX,
                               double fontScale = 0.5, int thickness = 1);
    
    static void drawTextWithBackground(cv::Mat& image, const std::string& text,
                                      const cv::Point& position,
                                      const cv::Scalar& textColor = cv::Scalar(255, 255, 255),
                                      const cv::Scalar& bgColor = cv::Scalar(0, 0, 0),
                                      double fontScale = 0.5, int thickness = 1);
    
    // Dashboard creation
    static cv::Mat createDashboard(const cv::Mat& mainView,
                                  const std::vector<cv::Mat>& additionalViews,
                                  const std::map<std::string, std::string>& metrics);
    
    // Video frame utilities
    static void addTimestamp(cv::Mat& image, const std::string& timestamp,
                            const cv::Point& position = cv::Point(-1, -1));
    
    static void addWatermark(cv::Mat& image, const std::string& watermark,
                            const cv::Point& position = cv::Point(-1, -1));
    
private:
    static const std::vector<cv::Scalar> CLASS_COLORS;
    static const std::map<CongestionLevel, cv::Scalar> CONGESTION_COLORS;
};

} // namespace traffic

#endif // VISUALIZATION_H
