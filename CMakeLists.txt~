cmake_minimum_required(VERSION 2.8) 
PROJECT (vidasa)
find_package(OpenCV REQUIRED )

set( NAME_SRC
src/main_processor.cpp
src/pedestrian_det.cpp
src/vehicle_det.cpp   			
)

set( NAME_HEADERS       
include/main_processor.hpp


)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include/ )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( vidasa ${NAME_SRC} ${NAME_HEADERS} )

target_link_libraries(vidasa ${OpenCV_LIBS} )
find_package(cvBlob REQUIRED)
target_link_libraries(vidasa ${cvBlob_LIBS})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
