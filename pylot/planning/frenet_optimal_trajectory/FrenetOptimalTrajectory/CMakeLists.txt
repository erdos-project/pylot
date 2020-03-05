cmake_minimum_required(VERSION 3.15)
project(FrenetOptimalTrajectory)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(Eigen3)
include_directories("${CMAKE_SOURCE_DIR}/src"
                    "${CMAKE_SOURCE_DIR}/src/CubicSpline"
                    "${CMAKE_SOURCE_DIR}/src/Polynomials"
                    "${CMAKE_SOURCE_DIR}/src/FrenetOptimalTrajectory")

add_library(FrenetOptimalTrajectory SHARED
            src/main.cpp
            src/utils.h
            src/constants.h
            src/Polynomials/QuarticPolynomial.cpp
            src/Polynomials/QuarticPolynomial.h
            src/Polynomials/QuinticPolynomial.cpp
            src/Polynomials/QuinticPolynomial.h
            src/CubicSpline/CubicSpline1D.cpp
            src/CubicSpline/CubicSpline1D.h
            src/CubicSpline/CubicSpline2D.cpp
            src/CubicSpline/CubicSpline2D.h
            src/FrenetOptimalTrajectory/FrenetOptimalTrajectory.cpp
            src/FrenetOptimalTrajectory/FrenetOptimalTrajectory.h
            src/FrenetOptimalTrajectory/FrenetPath.cpp
            src/FrenetOptimalTrajectory/FrenetPath.h
            src/FrenetOptimalTrajectory/fot_wrapper.cpp)
add_executable(FrenetOptimalTrajectoryTest
               src/main.cpp
               src/utils.h
               src/constants.h
               src/Polynomials/QuarticPolynomial.cpp
               src/Polynomials/QuarticPolynomial.h
               src/Polynomials/QuinticPolynomial.cpp
               src/Polynomials/QuinticPolynomial.h
               src/CubicSpline/CubicSpline1D.cpp
               src/CubicSpline/CubicSpline1D.h
               src/CubicSpline/CubicSpline2D.cpp
               src/CubicSpline/CubicSpline2D.h
               src/FrenetOptimalTrajectory/FrenetOptimalTrajectory.cpp
               src/FrenetOptimalTrajectory/FrenetOptimalTrajectory.h
               src/FrenetOptimalTrajectory/FrenetPath.cpp
               src/FrenetOptimalTrajectory/FrenetPath.h
               src/FrenetOptimalTrajectory/fot_wrapper.cpp)
target_link_libraries(FrenetOptimalTrajectory Eigen3::Eigen)
target_link_libraries(FrenetOptimalTrajectoryTest Eigen3::Eigen)