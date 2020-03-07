#ifndef FRENET_OPTIMAL_TRAJECTORY_UTILS_H
#define FRENET_OPTIMAL_TRAJECTORY_UTILS_H

#include <cmath>
#include <tuple>

inline double norm(double x, double y) {
    return sqrt(pow(x, 2) + pow(y, 2));
}

inline void as_unit_vector(std::tuple<double, double>& vec) {
    double magnitude = norm(std::get<0>(vec), std::get<1>(vec));
    if (magnitude > 0) {
        std::get<0>(vec) = std::get<0>(vec) / magnitude;
        std::get<1>(vec) = std::get<1>(vec) / magnitude;
    }
}

inline double dot(const std::tuple<double, double>& vec1,
                  const std::tuple<double, double>& vec2) {
    return std::get<0>(vec1) * std::get<0>(vec2) +
           std::get<1>(vec1) * std::get<1>(vec2);
}
#endif //FRENET_OPTIMAL_TRAJECTORY_UTILS_H
