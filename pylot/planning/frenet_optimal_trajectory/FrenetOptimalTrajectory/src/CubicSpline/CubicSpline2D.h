#ifndef FRENETOPTIMALTRAJECTORY_CUBICSPLINE2D_H
#define FRENETOPTIMALTRAJECTORY_CUBICSPLINE2D_H

#include <vector>
#include "CubicSpline1D.h"

// 2-dimensional cubic spline class.
// For technical details see: http://mathworld.wolfram.com/CubicSpline.html
class CubicSpline2D {
public:
    CubicSpline2D();
    CubicSpline2D(std::vector<double> &x, std::vector<double> &y);
    double calc_x(double t);
    double calc_y(double t);
    double calc_curvature(double t);
    double calc_yaw(double t);
    double find_s(double x, double y, double s0);

private:
    std::vector<double> s;
    CubicSpline1D sx, sy;
    static void calc_s(std::vector<double>& x, std::vector<double>& y,
                       std::vector<double>& result);
};

#endif //FRENETOPTIMALTRAJECTORY_CUBICSPLINE2D_H
