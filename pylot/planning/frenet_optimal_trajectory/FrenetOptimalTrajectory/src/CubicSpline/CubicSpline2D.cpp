#include "CubicSpline2D.h"
#include "utils.h"

#include <algorithm>
#include <numeric>
#include <cmath>

using namespace std;

// Default constructor
CubicSpline2D::CubicSpline2D() = default;

// Construct the 2-dimensional cubic spline
CubicSpline2D::CubicSpline2D(const std::vector<double> &x,
                             const std::vector<double> &y) {
    calc_s(x, y);
    sx = CubicSpline1D(s, x);
    sy = CubicSpline1D(s, y);
}

// Calculate the s values for interpolation given x, y
void CubicSpline2D::calc_s(const std::vector<double>& x,
                           const std::vector<double>& y) {
    int nx = x.size();
    vector<double> dx (nx);
    vector<double> dy (nx);
    adjacent_difference(x.begin(), x.end(), dx.begin());
    adjacent_difference(y.begin(), y.end(), dy.begin());
    dx.erase(dx.begin());
    dy.erase(dy.begin());

    double cum_sum = 0.0;
    s.push_back(cum_sum);
    for (int i = 0; i < nx - 1; i++) {
        cum_sum += norm(dx[i], dy[i]);
        s.push_back(cum_sum);
    }
    s.erase(unique(s.begin(), s.end()), s.end());
}

// Calculate the x position along the spline at given t
double CubicSpline2D::calc_x(double t) {
    return sx.calc_der0(t);
}

// Calculate the y position along the spline at given t
double CubicSpline2D::calc_y(double t) {
    return sy.calc_der0(t);
}

// Calculate the curvature along the spline at given t
double CubicSpline2D::calc_curvature(double t){
    double dx = sx.calc_der1(t);
    double ddx = sx.calc_der2(t);
    double dy = sy.calc_der1(t);
    double ddy = sy.calc_der2(t);
    double k = (ddy * dx - ddx * dy) /
            pow(pow(dx, 2) + pow(dy, 2), 1.5);
    return k;
}

// Calculate the yaw along the spline at given t
double CubicSpline2D::calc_yaw(double t) {
    double dx = sx.calc_der1(t);
    double dy = sy.calc_der1(t);
    double yaw = atan2(dy, dx);
    return yaw;
}

// Given x, y positions and an initial guess s0, find the closest s value
double CubicSpline2D::find_s(double x, double y, double s0) {
    double s_closest = s0;
    double closest = INFINITY;
    double si = s0;

    do {
        double px = calc_x(si);
        double py = calc_y(si);
        double dist = norm(x - px, y - py);
        if (dist < closest) {
            closest = dist;
            s_closest = si;
        }
        si += 0.1;
    } while (si < s0 + 10);
    return s_closest;
}