#include "FrenetOptimalTrajectory.h"
#include "QuarticPolynomial.h"
#include "QuinticPolynomial.h"
#include "constants.h"
#include "utils.h"

#include <cmath>
#include <iostream>
#include <utility>

using namespace std;

// Compute the frenet optimal trajectory
FrenetOptimalTrajectory::FrenetOptimalTrajectory(vector<double>& x_,
        vector<double>& y_, double s0_, double c_speed_, double c_d_,
        double c_d_d_, double c_d_dd_, double target_speed_,
        vector<tuple<double, double>>& obstacles_):
        x(x_), y(y_), s0(s0_), c_speed(c_speed_), c_d(c_d_), c_d_d(c_d_d_),
        c_d_dd(c_d_dd_), target_speed(target_speed_), obstacles(obstacles_) {
    csp = new CubicSpline2D(x, y);

    // calculate the trajectories
    calc_frenet_paths();

    // select the best path
    double mincost = INFINITY;
    for (FrenetPath* fp : frenet_paths) {
        if (fp->cf <= mincost) {
            mincost = fp->cf;
            best_frenet_path = fp;
        }
    }
}

FrenetOptimalTrajectory::~FrenetOptimalTrajectory() {
    delete csp;
    for (FrenetPath* fp : frenet_paths) {
        delete fp;
    }
}

// Return the best path
FrenetPath* FrenetOptimalTrajectory::getBestPath() {
    return best_frenet_path;
}

// Calculate frenet paths
void FrenetOptimalTrajectory::calc_frenet_paths() {
    double t, ti, tv, jp, js, ds;
    FrenetPath* fp, *tfp;

    double di = -MAX_ROAD_WIDTH;
    // generate path to each offset goal
    do {
        ti = MINT;
        // lateral motion planning
        do {
            ti += DT;
            fp = new FrenetPath();
            QuinticPolynomial lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di,
                    0.0, 0.0, ti);
            t = 0;
            // construct frenet path
            do {
                fp->t.push_back(t);
                fp->d.push_back(lat_qp.calc_point(t));
                fp->d_d.push_back(lat_qp.calc_first_derivative(t));
                fp->d_dd.push_back(lat_qp.calc_second_derivative(t));
                fp->d_ddd.push_back(lat_qp.calc_third_derivative(t));
                t += DT;
            } while (t < ti);

            // velocity keeping
            tv = target_speed - D_T_S * N_S_SAMPLE;
            do {
                jp = 0;
                js = 0;

                // copy frenet path
                tfp = new FrenetPath();
                for (double tt : fp->t) {
                    tfp->t.push_back(tt);
                    tfp->d.push_back(lat_qp.calc_point(tt));
                    tfp->d_d.push_back(lat_qp.calc_first_derivative(tt));
                    tfp->d_dd.push_back(lat_qp.calc_second_derivative(tt));
                    tfp->d_ddd.push_back(lat_qp.calc_third_derivative(tt));
                    jp += pow(lat_qp.calc_third_derivative(tt), 2);
                    // square jerk
                }
                QuarticPolynomial lon_qp = QuarticPolynomial(s0, c_speed,
                        0.0, tv, 0.0, ti);

                // longitudinal motion
                for (double tp : tfp->t) {
                    tfp->s.push_back(lon_qp.calc_point(tp));
                    tfp->s_d.push_back(lon_qp.calc_first_derivative(tp));
                    tfp->s_dd.push_back(lon_qp.calc_second_derivative(tp));
                    tfp->s_ddd.push_back(lon_qp.calc_third_derivative(tp));
                    js += pow(lon_qp.calc_third_derivative(tp), 2);
                    // square jerk
                }

                // calculate costs
                ds = pow(target_speed - tfp->s_d.back(), 2);
                tfp->cd = KJ * jp + KT * ti + KD * pow(tfp->d.back(), 2);
                tfp->cv = KJ * js + KT * ti + KD * ds;
                tfp->cf = KLAT * tfp->cd + KLON * tfp->cv;


                // append
                tfp->to_global_path(csp);
                if (!tfp->is_valid_path(obstacles)) {
                    // deallocate memory and continue
                    delete tfp;
                    tv += D_T_S;
                    continue;
                }
                frenet_paths.push_back(tfp);
                tv += D_T_S;
            } while (tv < target_speed + D_T_S * N_S_SAMPLE);
            // make sure to deallocate
            delete fp;
        } while(ti < MAXT);
        di += D_ROAD_W;
    } while (di < MAX_ROAD_WIDTH);
}

