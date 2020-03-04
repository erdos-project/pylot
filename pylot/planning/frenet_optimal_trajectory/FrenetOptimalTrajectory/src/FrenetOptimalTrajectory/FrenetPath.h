#ifndef FRENET_OPTIMAL_TRAJECTORY_FRENETPATH_H
#define FRENET_OPTIMAL_TRAJECTORY_FRENETPATH_H

#include <vector>

typedef struct  {
    std::vector<double> t;          // time
    std::vector<double> d;          // lateral offset
    std::vector<double> d_d;        // lateral speed
    std::vector<double> d_dd;       // lateral acceleration
    std::vector<double> d_ddd;      // lateral jerk
    std::vector<double> s;          // s position along spline
    std::vector<double> s_d;        // s speed
    std::vector<double> s_dd;       // s acceleration
    std::vector<double> s_ddd;      // s jerk

    std::vector<double> x;          // x position
    std::vector<double> y;          // y position
    std::vector<double> yaw;        // yaw in rad
    std::vector<double> ds;         // speed
    std::vector<double> c;          // curvature
    double cd = 0.0;                // lateral cost
    double cv = 0.0;                // longitudinal cost
    double cf = 0.0;                // final cost
} FrenetPath;

#endif //FRENET_OPTIMAL_TRAJECTORY_FRENETPATH_H
