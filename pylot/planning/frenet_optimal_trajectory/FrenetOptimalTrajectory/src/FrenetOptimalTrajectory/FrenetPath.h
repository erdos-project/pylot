#ifndef FRENET_OPTIMAL_TRAJECTORY_FRENETPATH_H
#define FRENET_OPTIMAL_TRAJECTORY_FRENETPATH_H

#include "CubicSpline2D.h"

#include <vector>
#include <tuple>

class FrenetPath {
public:
    // Frenet attributes
    std::vector<double> t;          // time
    std::vector<double> d;          // lateral offset
    std::vector<double> d_d;        // lateral speed
    std::vector<double> d_dd;       // lateral acceleration
    std::vector<double> d_ddd;      // lateral jerk
    std::vector<double> s;          // s position along spline
    std::vector<double> s_d;        // s speed
    std::vector<double> s_dd;       // s acceleration
    std::vector<double> s_ddd;      // s jerk

    // Euclidean attributes
    std::vector<double> x;          // x position
    std::vector<double> y;          // y position
    std::vector<double> yaw;        // yaw in rad
    std::vector<double> ds;         // speed
    std::vector<double> c;          // curvature

    // Cost attributes
    double cd = 0.0;                // lateral cost
    double cv = 0.0;                // longitudinal cost
    double cf = 0.0;                // final cost

    FrenetPath() = default;
    void to_global_path(CubicSpline2D* csp);
    bool is_valid_path(const std::vector<std::tuple<double, double>>& obstacles);
    bool is_collision(const std::vector<std::tuple<double, double>>& obstacles);
};

#endif //FRENET_OPTIMAL_TRAJECTORY_FRENETPATH_H
