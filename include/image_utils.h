#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace traffic {

class ImageUtils {
public:
    // Image loading and saving
    static cv::Mat loadImage(const std::string& path);
    static bool saveImage(const std::string& path, const cv::Mat& image);
    
    // Preprocessing
    static cv::Mat resize(const cv::Mat& image, int width, int height);
    static cv::Mat convertToGrayscale(const cv::Mat& image);
    static cv::Mat normalizeImage(const cv::Mat& image);
    static cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize = 5);
    static cv::Mat applyMedianBlur(const cv::Mat& image, int kernelSize = 5);
    static cv::Mat applyBilateralFilter(const cv::Mat& image);
    
    // Edge detection
    static cv::Mat cannyEdgeDetection(const cv::Mat& image, double threshold1 = 50, double threshold2 = 150);
    static cv::Mat sobelEdgeDetection(const cv::Mat& image);
    
    // Morphological operations
    static cv::Mat morphologicalOpen(const cv::Mat& image, int kernelSize = 5);
    static cv::Mat morphologicalClose(const cv::Mat& image, int kernelSize = 5);
    static cv::Mat dilate(const cv::Mat& image, int iterations = 1);
    static cv::Mat erode(const cv::Mat& image, int iterations = 1);
    
    // Color space conversions
    static cv::Mat convertToHSV(const cv::Mat& image);
    static cv::Mat convertToLAB(const cv::Mat& image);
    
    // Thresholding
    static cv::Mat binaryThreshold(const cv::Mat& image, double threshold = 127);
    static cv::Mat adaptiveThreshold(const cv::Mat& image);
    static cv::Mat otsuThreshold(const cv::Mat& image);
    
    // Region of Interest
    static cv::Mat extractROI(const cv::Mat& image, const cv::Rect& roi);
    static void drawROI(cv::Mat& image, const cv::Rect& roi, const cv::Scalar& color = cv::Scalar(0, 255, 0));
    
    // Batch operations
    static std::vector<cv::Mat> loadImagesFromDirectory(const std::string& directory);
    static void saveImagesToDirectory(const std::vector<cv::Mat>& images, const std::string& directory, const std::string& prefix = "img");
    
    // Utility functions
    static void displayImage(const std::string& windowName, const cv::Mat& image);
    static cv::Mat concatenateImages(const std::vector<cv::Mat>& images, bool horizontal = true);
    static void drawText(cv::Mat& image, const std::string& text, const cv::Point& position, const cv::Scalar& color = cv::Scalar(255, 255, 255));
    
    // Image quality metrics
    static double calculateMSE(const cv::Mat& img1, const cv::Mat& img2);
    static double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2);
    static double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2);
};

} // namespace traffic

#endif // IMAGE_UTILS_H
