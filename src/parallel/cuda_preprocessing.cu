/**
 * CUDA Image Preprocessing
 * GPU-accelerated image preprocessing for traffic analysis
 */

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// CUDA kernel for grayscale conversion
__global__ void rgbToGrayscaleKernel(unsigned char* input, unsigned char* output, 
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgbIdx = idx * 3;
        
        // Grayscale conversion: 0.299*R + 0.587*G + 0.114*B
        output[idx] = static_cast<unsigned char>(
            0.299f * input[rgbIdx + 2] +  // R
            0.587f * input[rgbIdx + 1] +  // G
            0.114f * input[rgbIdx]        // B
        );
    }
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(unsigned char* input, unsigned char* output,
                                   int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        float weightSum = 0.0f;
        int halfKernel = kernelSize / 2;
        
        for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
            for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    // Gaussian weight
                    float weight = expf(-(kx*kx + ky*ky) / (2.0f * 1.5f * 1.5f));
                    sum += input[ny * width + nx] * weight;
                    weightSum += weight;
                }
            }
        }
        
        output[y * width + x] = static_cast<unsigned char>(sum / weightSum);
    }
}

// CUDA kernel for edge detection (Sobel)
__global__ void sobelEdgeKernel(unsigned char* input, unsigned char* output,
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel
        int gx = -input[(y-1)*width + (x-1)] + input[(y-1)*width + (x+1)]
                -2*input[y*width + (x-1)] + 2*input[y*width + (x+1)]
                -input[(y+1)*width + (x-1)] + input[(y+1)*width + (x+1)];
        
        // Sobel Y kernel
        int gy = -input[(y-1)*width + (x-1)] - 2*input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                +input[(y+1)*width + (x-1)] + 2*input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
        
        // Magnitude
        int magnitude = static_cast<int>(sqrtf(gx*gx + gy*gy));
        output[y * width + x] = min(255, magnitude);
    }
}

// CUDA kernel for normalization
__global__ void normalizeKernel(unsigned char* input, unsigned char* output,
                                int width, int height, float minVal, float maxVal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float normalized = (input[idx] - minVal) / (maxVal - minVal) * 255.0f;
        output[idx] = static_cast<unsigned char>(max(0.0f, min(255.0f, normalized)));
    }
}

class CUDAImagePreprocessor {
public:
    CUDAImagePreprocessor() {
        // Check CUDA availability
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            cudaAvailable = false;
        } else {
            cudaAvailable = true;
            cudaSetDevice(0);
            
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "Using CUDA device: " << prop.name << std::endl;
        }
    }
    
    cv::Mat rgbToGrayscale(const cv::Mat& input) {
        if (!cudaAvailable) {
            std::cerr << "CUDA not available!" << std::endl;
            return cv::Mat();
        }
        
        int width = input.cols;
        int height = input.rows;
        
        // Allocate device memory
        unsigned char *d_input, *d_output;
        size_t inputSize = width * height * 3;
        size_t outputSize = width * height;
        
        cudaMalloc(&d_input, inputSize);
        cudaMalloc(&d_output, outputSize);
        
        // Copy input to device
        cudaMemcpy(d_input, input.data, inputSize, cudaMemcpyHostToDevice);
        
        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        
        // Copy result back
        cv::Mat output(height, width, CV_8UC1);
        cudaMemcpy(output.data, d_output, outputSize, cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        
        return output;
    }
    
    cv::Mat gaussianBlur(const cv::Mat& input, int kernelSize = 5) {
        if (!cudaAvailable) return cv::Mat();
        
        int width = input.cols;
        int height = input.rows;
        
        unsigned char *d_input, *d_output;
        size_t size = width * height;
        
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        
        cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, kernelSize);
        
        cv::Mat output(height, width, CV_8UC1);
        cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        return output;
    }
    
    cv::Mat sobelEdgeDetection(const cv::Mat& input) {
        if (!cudaAvailable) return cv::Mat();
        
        int width = input.cols;
        int height = input.rows;
        
        unsigned char *d_input, *d_output;
        size_t size = width * height;
        
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
        
        cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
        
        sobelEdgeKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        
        cv::Mat output(height, width, CV_8UC1);
        cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_input);
        cudaFree(d_output);
        
        return output;
    }
    
private:
    bool cudaAvailable;
};

int main(int argc, char** argv) {
    std::cout << "🚀 CUDA Image Preprocessor" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }
    
    // Load image
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << argv[1] << std::endl;
        return 1;
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    
    CUDAImagePreprocessor preprocessor;
    
    // Grayscale conversion
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat gray = preprocessor.rgbToGrayscale(image);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Grayscale conversion: " << duration.count() << " ms" << std::endl;
    cv::imwrite("output/cuda_grayscale.jpg", gray);
    
    // Gaussian blur
    start = std::chrono::high_resolution_clock::now();
    cv::Mat blurred = preprocessor.gaussianBlur(gray, 5);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Gaussian blur: " << duration.count() << " ms" << std::endl;
    cv::imwrite("output/cuda_blurred.jpg", blurred);
    
    // Edge detection
    start = std::chrono::high_resolution_clock::now();
    cv::Mat edges = preprocessor.sobelEdgeDetection(gray);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Edge detection: " << duration.count() << " ms" << std::endl;
    cv::imwrite("output/cuda_edges.jpg", edges);
    
    std::cout << "✅ Processing completed!" << std::endl;
    
    return 0;
}
