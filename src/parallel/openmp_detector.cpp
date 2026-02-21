/**
 * OpenMP Parallel Vehicle Detector
 * Uses OpenMP for multi-threaded image processing
 */

#include <omp.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include "image_utils.h"
#include "vehicle_detector.h"
#include "density_estimator.h"

namespace fs = std::filesystem;
using namespace traffic;

struct ImageResult {
    std::string filename;
    int vehicleCount;
    double processingTime;
    std::string congestionLevel;
};

int main(int argc, char** argv) {
    std::string input_dir = "data/images/test";
    std::string output_dir = "output/results";
    int num_threads = omp_get_max_threads();
    
    if (argc > 1) {
        input_dir = argv[1];
    }
    if (argc > 2) {
        output_dir = argv[2];
    }
    if (argc > 3) {
        num_threads = std::stoi(argv[3]);
    }
    
    std::cout << "🚀 OpenMP Parallel Vehicle Detector" << std::endl;
    std::cout << "   Input directory: " << input_dir << std::endl;
    std::cout << "   Output directory: " << output_dir << std::endl;
    std::cout << "   Number of threads: " << num_threads << std::endl;
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Get all image files
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }
    
    std::cout << "   Total images: " << imageFiles.size() << std::endl;
    
    if (imageFiles.empty()) {
        std::cerr << "❌ No images found in " << input_dir << std::endl;
        return 1;
    }
    
    // Create output directory
    fs::create_directories(output_dir);
    
    // Results storage
    std::vector<ImageResult> results(imageFiles.size());
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Parallel processing
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        // Each thread has its own detector and estimator
        VehicleDetector detector;
        DensityEstimator estimator(cv::Size(1920, 1080));
        
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < imageFiles.size(); ++i) {
            auto imgStartTime = std::chrono::high_resolution_clock::now();
            
            // Load image
            cv::Mat image = ImageUtils::loadImage(imageFiles[i]);
            
            if (image.empty()) {
                #pragma omp critical
                {
                    std::cerr << "⚠️  Thread " << thread_id 
                             << ": Failed to load " << imageFiles[i] << std::endl;
                }
                continue;
            }
            
            // Detect vehicles
            std::vector<Detection> detections = detector.detectVehicles(image);
            
            // Estimate density
            DensityMetrics metrics = estimator.estimateDensity(image, detections);
            
            // Draw results
            cv::Mat resultImage = detector.drawDetections(image, detections);
            resultImage = estimator.drawDensityMetrics(resultImage, metrics);
            
            // Save result
            std::string outputPath = output_dir + "/" + 
                                    fs::path(imageFiles[i]).filename().string();
            ImageUtils::saveImage(outputPath, resultImage);
            
            auto imgEndTime = std::chrono::high_resolution_clock::now();
            auto imgDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                imgEndTime - imgStartTime);
            
            // Store results
            results[i].filename = fs::path(imageFiles[i]).filename().string();
            results[i].vehicleCount = metrics.vehicleCount;
            results[i].processingTime = imgDuration.count() / 1000.0;
            
            switch (metrics.congestionLevel) {
                case CongestionLevel::FREE_FLOW:
                    results[i].congestionLevel = "FREE_FLOW";
                    break;
                case CongestionLevel::LIGHT:
                    results[i].congestionLevel = "LIGHT";
                    break;
                case CongestionLevel::MODERATE:
                    results[i].congestionLevel = "MODERATE";
                    break;
                case CongestionLevel::HEAVY:
                    results[i].congestionLevel = "HEAVY";
                    break;
                case CongestionLevel::SEVERE:
                    results[i].congestionLevel = "SEVERE";
                    break;
            }
            
            // Progress update
            #pragma omp critical
            {
                if ((i + 1) % 10 == 0 || (i + 1) == imageFiles.size()) {
                    std::cout << "Thread " << thread_id << ": Processed " 
                             << (i + 1) << "/" << imageFiles.size() 
                             << " images" << std::endl;
                }
            }
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime);
    
    // Calculate statistics
    double totalProcessingTime = 0;
    int totalVehicles = 0;
    
    for (const auto& result : results) {
        totalProcessingTime += result.processingTime;
        totalVehicles += result.vehicleCount;
    }
    
    double avgProcessingTime = totalProcessingTime / results.size();
    double avgVehicles = static_cast<double>(totalVehicles) / results.size();
    double speedup = totalProcessingTime / (totalDuration.count() / 1000.0);
    
    // Save results to CSV
    std::string csvPath = output_dir + "/processing_results.csv";
    std::ofstream csvFile(csvPath);
    csvFile << "filename,vehicle_count,processing_time,congestion_level\\n";
    
    for (const auto& result : results) {
        csvFile << result.filename << ","
               << result.vehicleCount << ","
               << result.processingTime << ","
               << result.congestionLevel << "\\n";
    }
    
    csvFile.close();
    
    // Print summary
    std::cout << "\n✅ Processing completed!" << std::endl;
    std::cout << "   Total time: " << totalDuration.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "   Images processed: " << results.size() << std::endl;
    std::cout << "   Average processing time: " << avgProcessingTime << " seconds/image" << std::endl;
    std::cout << "   Average vehicles per image: " << avgVehicles << std::endl;
    std::cout << "   Speedup: " << speedup << "x" << std::endl;
    std::cout << "   Efficiency: " << (speedup / num_threads * 100) << "%" << std::endl;
    std::cout << "   Results saved to: " << output_dir << std::endl;
    std::cout << "   CSV report: " << csvPath << std::endl;
    
    return 0;
}
