#include "QuinticPolynomial.h"

#include <Eigen/LU>
#include <cmath>

using namespace Eigen;

QuinticPolynomial::QuinticPolynomial(double xs, double vxs, double axs,
        double xe, double vxe, double axe, double t):
        a0(xs), a1(vxs) {
    a2 = axs / 2.0;
    Matrix3d A;
    Vector3d B;
    A << pow(t, 3), pow(t, 4), pow(t, 5), 3 * pow(t, 2),
    4 * pow(t, 3), 5 * pow(t, 4), 6 * t, 12 * pow(t, 2),
    20 * pow(t, 3);
    B << xe - a0 - a1 * t - a2 * pow(t, 2), vxe - a1 - 2 * a2 * t,
    axe - 2 * a2;
    Matrix3d A_inv = A.inverse();
    Vector3d x = A_inv * B;
    a3 = x[0];
    a4 = x[1];
    a5 = x[2];
}

double QuinticPolynomial::calc_point(double t) {
    return a0 + a1 * t + a2 * pow(t, 2) + a3 * pow(t, 3) +
    a4 * pow(t, 4) + a5 * pow(t, 5);
}

double QuinticPolynomial::calc_first_derivative(double t) {
    return a1 + 2 * a2 * t + 3 * a3 * pow(t, 2) + 4 * a4 * pow(t, 3) +
    5 * a5 * pow(t, 4);
}

double QuinticPolynomial::calc_second_derivative(double t) {
    return 2 * a2 + 6 * a3 * t + 12 * a4 * pow(t, 2) + 20 * a5 * pow(t, 3);
}

double QuinticPolynomial::calc_third_derivative(double t) {
    return 6 * a3 + 24 * a4 * t + 60 * a5 * pow(t, 2);
}