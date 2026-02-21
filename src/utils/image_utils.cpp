#include "image_utils.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace traffic {

cv::Mat ImageUtils::loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << path << std::endl;
    }
    return image;
}

bool ImageUtils::saveImage(const std::string& path, const cv::Mat& image) {
    return cv::imwrite(path, image);
}

cv::Mat ImageUtils::resize(const cv::Mat& image, int width, int height) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height));
    return resized;
}

cv::Mat ImageUtils::convertToGrayscale(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    return gray;
}

cv::Mat ImageUtils::normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX);
    return normalized;
}

cv::Mat ImageUtils::applyGaussianBlur(const cv::Mat& image, int kernelSize) {
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(kernelSize, kernelSize), 0);
    return blurred;
}

cv::Mat ImageUtils::applyMedianBlur(const cv::Mat& image, int kernelSize) {
    cv::Mat blurred;
    cv::medianBlur(image, blurred, kernelSize);
    return blurred;
}

cv::Mat ImageUtils::applyBilateralFilter(const cv::Mat& image) {
    cv::Mat filtered;
    cv::bilateralFilter(image, filtered, 9, 75, 75);
    return filtered;
}

cv::Mat ImageUtils::cannyEdgeDetection(const cv::Mat& image, double threshold1, double threshold2) {
    cv::Mat edges;
    cv::Mat gray = convertToGrayscale(image);
    cv::Canny(gray, edges, threshold1, threshold2);
    return edges;
}

cv::Mat ImageUtils::sobelEdgeDetection(const cv::Mat& image) {
    cv::Mat gray = convertToGrayscale(image);
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, edges;
    
    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
    
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);
    return edges;
}

cv::Mat ImageUtils::morphologicalOpen(const cv::Mat& image, int kernelSize) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(image, result, cv::MORPH_OPEN, kernel);
    return result;
}

cv::Mat ImageUtils::morphologicalClose(const cv::Mat& image, int kernelSize) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(image, result, cv::MORPH_CLOSE, kernel);
    return result;
}

cv::Mat ImageUtils::dilate(const cv::Mat& image, int iterations) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(image, result, kernel, cv::Point(-1, -1), iterations);
    return result;
}

cv::Mat ImageUtils::erode(const cv::Mat& image, int iterations) {
    cv::Mat result;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(image, result, kernel, cv::Point(-1, -1), iterations);
    return result;
}

cv::Mat ImageUtils::convertToHSV(const cv::Mat& image) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    return hsv;
}

cv::Mat ImageUtils::convertToLAB(const cv::Mat& image) {
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    return lab;
}

cv::Mat ImageUtils::binaryThreshold(const cv::Mat& image, double threshold) {
    cv::Mat binary;
    cv::Mat gray = convertToGrayscale(image);
    cv::threshold(gray, binary, threshold, 255, cv::THRESH_BINARY);
    return binary;
}

cv::Mat ImageUtils::adaptiveThreshold(const cv::Mat& image) {
    cv::Mat binary;
    cv::Mat gray = convertToGrayscale(image);
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    return binary;
}

cv::Mat ImageUtils::otsuThreshold(const cv::Mat& image) {
    cv::Mat binary;
    cv::Mat gray = convertToGrayscale(image);
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return binary;
}

cv::Mat ImageUtils::extractROI(const cv::Mat& image, const cv::Rect& roi) {
    return image(roi).clone();
}

void ImageUtils::drawROI(cv::Mat& image, const cv::Rect& roi, const cv::Scalar& color) {
    cv::rectangle(image, roi, color, 2);
}

std::vector<cv::Mat> ImageUtils::loadImagesFromDirectory(const std::string& directory) {
    std::vector<cv::Mat> images;
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                cv::Mat img = loadImage(entry.path().string());
                if (!img.empty()) {
                    images.push_back(img);
                }
            }
        }
    }
    
    return images;
}

void ImageUtils::saveImagesToDirectory(const std::vector<cv::Mat>& images, 
                                      const std::string& directory, 
                                      const std::string& prefix) {
    fs::create_directories(directory);
    
    for (size_t i = 0; i < images.size(); ++i) {
        std::string filename = directory + "/" + prefix + "_" + std::to_string(i) + ".jpg";
        saveImage(filename, images[i]);
    }
}

void ImageUtils::displayImage(const std::string& windowName, const cv::Mat& image) {
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, image);
    cv::waitKey(0);
    cv::destroyWindow(windowName);
}

cv::Mat ImageUtils::concatenateImages(const std::vector<cv::Mat>& images, bool horizontal) {
    if (images.empty()) return cv::Mat();
    
    cv::Mat result;
    if (horizontal) {
        cv::hconcat(images, result);
    } else {
        cv::vconcat(images, result);
    }
    return result;
}

void ImageUtils::drawText(cv::Mat& image, const std::string& text, 
                         const cv::Point& position, const cv::Scalar& color) {
    cv::putText(image, text, position, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}

double ImageUtils::calculateMSE(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    
    cv::Scalar s = cv::sum(diff);
    double sse = s.val[0] + s.val[1] + s.val[2];
    double mse = sse / (double)(img1.channels() * img1.total());
    
    return mse;
}

double ImageUtils::calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    double mse = calculateMSE(img1, img2);
    if (mse <= 1e-10) return 0;
    
    double psnr = 10.0 * log10((255 * 255) / mse);
    return psnr;
}

double ImageUtils::calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;
    
    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);
    
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);
    
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    
    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    
    cv::Scalar mssim = cv::mean(ssim_map);
    return mssim.val[0];
}

} // namespace traffic
