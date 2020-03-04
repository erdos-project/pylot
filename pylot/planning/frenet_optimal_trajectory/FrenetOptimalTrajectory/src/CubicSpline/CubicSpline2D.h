#ifndef FRENET_OPTIMAL_TRAJECTORY_CUBICSPLINE2D_H
#define FRENET_OPTIMAL_TRAJECTORY_CUBICSPLINE2D_H

#include "CubicSpline1D.h"

#include <vector>

// 2-dimensional cubic spline class.
// For technical details see: http://mathworld.wolfram.com/CubicSpline.html
class CubicSpline2D {
public:
    CubicSpline2D();
    CubicSpline2D(const std::vector<double> &x, const std::vector<double> &y);
    double calc_x(double t);
    double calc_y(double t);
    double calc_curvature(double t);
    double calc_yaw(double t);
    double find_s(double x, double y, double s0);

private:
    std::vector<double> s;
    CubicSpline1D sx, sy;
    void calc_s(const std::vector<double>& x,
                const std::vector<double>& y);
};

#endif //FRENET_OPTIMAL_TRAJECTORY_CUBICSPLINE2D_H
