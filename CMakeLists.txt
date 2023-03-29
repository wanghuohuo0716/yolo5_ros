cmake_minimum_required(VERSION 2.6)

project(yolov5)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")


find_package(CUDA REQUIRED)

find_package(catkin REQUIRED
        COMPONENTS
            roscpp
            nav_msgs
            sensor_msgs
            std_msgs
            image_transport
	    message_generation
	    cv_bridge
            tf2
        )

#find_package(cv_bridge)
#include_directories(${cv_bridge_INCLUDE_DIRS})


find_package(OpenCV 4)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include ${catkin_INCLUDE_DIRS})
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()



add_message_files(
    
    FILES 
    single_res.msg 
    multi_res.msg 
    yolo_detection.msg

)

generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
)

catkin_package(
    CATKIN_DEPENDS
        roscpp
        nav_msgs
        image_transport
        std_msgs
        cv_bridge
        sensor_msgs
	message_runtime
        tf2
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)



message("${OpenCV_INCLUDE_DIRS}")

add_executable(yolov5s ${PROJECT_SOURCE_DIR}/yolov5s.cpp ${PROJECT_SOURCE_DIR}/videoCapture.cpp)
target_link_libraries(yolov5s nvinfer)
target_link_libraries(yolov5s cudart)
target_link_libraries(yolov5s yololayer)
target_link_libraries(yolov5s ${OpenCV_LIBRARIES})
target_link_libraries(yolov5s ${catkin_LIBRARIES})
#target_link_libraries(yolov5s ${cv_bridge_LIBRARIES})

add_definitions(-O2 -pthread)

