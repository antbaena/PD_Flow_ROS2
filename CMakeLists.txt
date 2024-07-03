cmake_minimum_required(VERSION 3.0)
project(PD-Flow)

# Asegúrate de que CUDA esté disponible
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-arch=sm_61")

# Construir la biblioteca CUDA
cuda_add_library(pdflow_cudalib pdflow_cudalib.h pdflow_cudalib.cu)

set(DEFAULT_BUILD_EVALUATOR ON)
set(BUILD_EVALUATOR ${DEFAULT_BUILD_EVALUATOR} CACHE BOOL "Build the scene flow evaluator for a RGB-D frame pair that uses OpenCV")

# Buscar OpenCV
if (BUILD_EVALUATOR)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    add_executable(Scene-Flow-Impair
            main_scene_flow_impair.cpp
            scene_flow_impair.cpp
            scene_flow_impair.h)

    target_link_libraries(Scene-Flow-Impair
            ${OpenCV_LIBS}
            ${CUDA_LIBRARIES}
            pdflow_cudalib)
endif (BUILD_EVALUATOR)

set(DEFAULT_BUILD_RT_VISUALIZATION OFF)  # Si no necesitas visualización en tiempo real
set(BUILD_RT_VISUALIZATION ${DEFAULT_BUILD_RT_VISUALIZATION} CACHE BOOL "Build the scene flow estimator with real-time visualization that requires OpenNI2")

# Si no se necesita visualización en tiempo real, no construyas esta parte
if (BUILD_RT_VISUALIZATION)
    find_package(OpenNI REQUIRED)
    set(OpenNI2_libdir "/usr/lib")
    include_directories("/usr/include/openni2")
    link_directories(${OpenNI2_libdir})
    set(OpenNI_lib "${OpenNI2_libdir}/libOpenNI2.so")

    add_executable(Scene-Flow-Visualization
            main_scene_flow_visualization.cpp
            scene_flow_visualization.cpp
            scene_flow_visualization.h
            legend_pdflow.xpm)

    target_link_libraries(Scene-Flow-Visualization
            ${OpenCV_LIBS}
            ${CUDA_LIBRARIES}
            ${OpenNI_lib}
            pdflow_cudalib)
endif (BUILD_RT_VISUALIZATION)

# Configurar optimización de compilación
if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
endif(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
