cmake_minimum_required(VERSION 3.15)
project(SmartTrafficHPC LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -march=native")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_75")

# Source files
set(SOURCES
    src/generate_data.cpp
    src/image_processing/vehicle_detector.cpp
    src/image_processing/density_estimator.cpp
    src/image_processing/speed_tracker.cpp
    src/image_processing/congestion_analyzer.cpp
    src/utils/image_utils.cpp
    src/utils/visualization.cpp
)

# Parallel processing sources
set(PARALLEL_SOURCES
    src/parallel/mpi_image_batch.cpp
    src/parallel/openmp_detector.cpp
)

# CUDA sources
set(CUDA_SOURCES
    src/parallel/cuda_preprocessing.cu
)

# Main executable
add_executable(generate_data src/generate_data.cpp)

# Vehicle detector executable
add_executable(vehicle_detector 
    src/image_processing/vehicle_detector.cpp
    src/utils/image_utils.cpp
)
target_link_libraries(vehicle_detector ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Density estimator executable
add_executable(density_estimator
    src/image_processing/density_estimator.cpp
    src/utils/image_utils.cpp
)
target_link_libraries(density_estimator ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Speed tracker executable
add_executable(speed_tracker
    src/image_processing/speed_tracker.cpp
    src/utils/image_utils.cpp
)
target_link_libraries(speed_tracker ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Congestion analyzer executable
add_executable(congestion_analyzer
    src/image_processing/congestion_analyzer.cpp
    src/utils/image_utils.cpp
    src/utils/visualization.cpp
)
target_link_libraries(congestion_analyzer ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# MPI batch processor
add_executable(mpi_batch_processor
    src/parallel/mpi_image_batch.cpp
    src/utils/image_utils.cpp
)
target_link_libraries(mpi_batch_processor ${OpenCV_LIBS} ${MPI_LIBRARIES} OpenMP::OpenMP_CXX)

# OpenMP detector
add_executable(openmp_detector
    src/parallel/openmp_detector.cpp
    src/utils/image_utils.cpp
)
target_link_libraries(openmp_detector ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# CUDA preprocessor
cuda_add_executable(cuda_preprocessor
    src/parallel/cuda_preprocessing.cu
)
target_link_libraries(cuda_preprocessor ${OpenCV_LIBS} ${CUDA_LIBRARIES})

# Main traffic monitoring application
add_executable(main_traffic_monitor
    src/main_traffic_monitor.cpp
    src/image_processing/vehicle_detector.cpp
    src/image_processing/density_estimator.cpp
    src/image_processing/speed_tracker.cpp
    src/image_processing/congestion_analyzer.cpp
    src/utils/image_utils.cpp
    src/utils/visualization.cpp
)
target_link_libraries(main_traffic_monitor ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Installation
install(TARGETS 
    generate_data 
    vehicle_detector 
    density_estimator 
    speed_tracker 
    congestion_analyzer
    mpi_batch_processor
    openmp_detector
    cuda_preprocessor
    main_traffic_monitor
    DESTINATION bin
)
