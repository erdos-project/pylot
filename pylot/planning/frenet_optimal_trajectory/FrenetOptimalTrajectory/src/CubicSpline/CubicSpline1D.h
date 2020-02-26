#ifndef FRENETOPTIMALTRAJECTORY_CUBICSPLINE1D_H
#define FRENETOPTIMALTRAJECTORY_CUBICSPLINE1D_H

#include <vector>
#include <Eigen/LU>

// 1-dimensional cubic spline class.
// For technical details see: http://mathworld.wolfram.com/CubicSpline.html
class CubicSpline1D {
public:
    int nx{};
    CubicSpline1D();
    CubicSpline1D (std::vector<double>& v1, std::vector<double>& v2);
    double calc_der0 (double t);
    double calc_der1 (double t);
    double calc_der2 (double t);
private:
    std::vector<double> a, b, c, d, w, x, y;
    int search_index (double t);
    void matrix_a (std::vector<double>& deltas, Eigen::MatrixXd& result);
    void vector_b (std::vector<double>& deltas, Eigen::VectorXd& result);
};

#endif //FRENETOPTIMALTRAJECTORY_CUBICSPLINE1D_H
