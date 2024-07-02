cmake_minimum_required(VERSION 3.0)
project(PD-Flow)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-arch=sm_61")

# Build scene flow CUDA lib
cuda_add_library(pdflow_cudalib pdflow_cudalib.h pdflow_cudalib.cu)

set(DEFAULT_BUILD_EVALUATOR ON)
set(BUILD_EVALUATOR ${DEFAULT_BUILD_EVALUATOR} CACHE BOOL "Build the scene flow evaluator for a RGB-D frame pair that uses OpenCV")

# Find OpenCV and Eigen
if (BUILD_EVALUATOR)
    find_package(OpenCV REQUIRED)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
    include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

    add_executable(Scene-Flow-Impair  
        main_scene_flow_impair.cpp
        scene_flow_impair.cpp
        scene_flow_impair.h)
        
    target_link_libraries(Scene-Flow-Impair 
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        pdflow_cudalib)
endif (BUILD_EVALUATOR)

set(DEFAULT_BUILD_RT_VISUALIZATION ON)
set(BUILD_RT_VISUALIZATION ${DEFAULT_BUILD_RT_VISUALIZATION} CACHE BOOL "Build the scene flow estimator with real-time visualization that requires MRPT and OpenNI2")

# Build real-time scene flow with visualization
if (BUILD_RT_VISUALIZATION)
    find_package(MRPT REQUIRED base gui opengl)
    
    if (WIN32)
        include_directories($ENV{OPENNI2_INCLUDE})
        link_directories($ENV{OPENNI2_LIB})
        set(OpenNI_lib "$ENV{OPENNI2_LIB}/OpenNI2.lib")
    endif (WIN32)
    
    if (UNIX)
        set(OpenNI2_libdir "/usr/lib")
        include_directories("/usr/include/openni2")
        link_directories(${OpenNI2_libdir})
        set(OpenNI_lib "${OpenNI2_libdir}/libOpenNI2.so")
    endif (UNIX)

    add_executable(Scene-Flow-Visualization  
        main_scene_flow_visualization.cpp
        scene_flow_visualization.cpp
        scene_flow_visualization.h
        legend_pdflow.xpm)
        
    target_link_libraries(Scene-Flow-Visualization
        ${MRPT_LIBS}
        ${CUDA_LIBRARIES}
        ${OpenNI_lib}
        pdflow_cudalib)
endif (BUILD_RT_VISUALIZATION)

# Set optimized building:
if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
endif(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
