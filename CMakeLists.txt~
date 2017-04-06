cmake_minimum_required(VERSION 2.8)
project(dvo)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
SET(CMAKE_BUILD_TYPE Debug)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )
include_directories(${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/thirdparty
)

find_package(OpenCV 2.4 REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# find pcl
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
ADD_DEFINITIONS(${PLC_DEFINITIONS})

add_library(imgprolib
src/FrameData.cpp
include/FrameData.h
src/viewer.cpp
include/viewer.h
)
target_link_libraries(imgprolib ${OpenCV_LIBS} ${PCL_LIBRARIES})

add_library(dvocorelib src/CalWeight.cpp
src/TransformEstimate.cpp
src/Least_squares.cpp

include/rgbdread.h
include/CalWeight.h
include/TransformEstimate.h
include/Least_squares.h
)

#target_link_libraries(dvocorelib
#imgprolib
#${PROJECT_SOURCE_DIR}/thirdparty/sophus_src/build/libSophus.so
#)

# sophus 
FIND_PACKAGE(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
target_link_libraries(dvocorelib
imgprolib
${Sophus_LIBRARIES}
)

add_executable(dvotest src/dvotest.cpp)
target_link_libraries(dvotest dvocorelib)
