#include "FrenetOptimalTrajectory.h"
#include "constants.h"
#include "QuarticPolynomial.h"
#include "QuinticPolynomial.h"
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
        x(x_), y(y_), s0(s0_),
        c_speed(c_speed_), c_d(c_d_), c_d_d(c_d_d_), c_d_dd(c_d_dd_),
        target_speed(target_speed_), obstacles(obstacles_) {

    csp = CubicSpline2D(x, y);

    // calculate the trajectories
    calc_frenet_paths();
    calc_global_paths();
    validate_paths();

    // select the best path
    double mincost = INFINITY;
    for (FrenetPath* fp : result_frenet_paths) {
        if (fp->cf <= mincost) {
            mincost = fp->cf;
            best_frenet_path = *fp;
        }
    }
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
                frenet_paths.push_back(tfp);
                tv += D_T_S;
            } while (tv < target_speed + D_T_S * N_S_SAMPLE);
        } while(ti < MAXT);
        di += D_ROAD_W;
    } while (di < MAX_ROAD_WIDTH);
}

// Convert the frenet paths to global paths in terms of x, y, yaw, velocity
void FrenetOptimalTrajectory::calc_global_paths() {
    double ix, iy, iyaw, di, fx, fy, dx, dy;
    for (FrenetPath* fp : frenet_paths) {
        // calc global positions
        for (int i = 0; i < fp->s.size(); i++) {
            ix = csp.calc_x(fp->s[i]);
            iy = csp.calc_y(fp->s[i]);
            if (isnan(ix) || isnan(iy)) break;

            iyaw = csp.calc_yaw(fp->s[i]);
            di = fp->d[i];
            fx = ix + di * cos(iyaw + M_PI / 2.0);
            fy = iy + di * sin(iyaw + M_PI / 2.0);
            fp->x.push_back(fx);
            fp->y.push_back(fy);
        }

        // calc yaw and ds
        for (int i = 0; i < fp->x.size() - 1; i++) {
            dx = fp->x[i+1] - fp->x[i];
            dy = fp->y[i+1] - fp->y[i];
            fp->yaw.push_back(atan2(dy, dx));
            fp->ds.push_back(hypot(dx, dy));
        }
        fp->yaw.push_back(fp->yaw.back());
        fp->ds.push_back(fp->ds.back());

        // calc curvature
        for (int i = 0; i < fp->yaw.size() - 1; i++) {
            fp->c.push_back((fp->yaw[i+1] - fp->yaw[i]) / fp->ds[i]);
        }
    }
}

// Validate the calculated frenet paths against threshold speed, acceleration,
// curvature and collision checks
void FrenetOptimalTrajectory::validate_paths() {
    // check all paths for validity
    for (FrenetPath* fp : frenet_paths) {
        // max speed check
        if (any_of(fp->s_d.begin(), fp->s_d.end(),
                [](int i){return abs(i) > MAX_SPEED;})) continue;
        // max accel check
        else if (any_of(fp->s_dd.begin(), fp->s_dd.end(),
                [](int i){return abs(i) > MAX_ACCEL;})) continue;
        // max curvature check
        else if (any_of(fp->c.begin(), fp->c.end(),
                [](int i){return abs(i) > MAX_CURVATURE;})) continue;
        // collision check
        else if (is_collision(*fp)) continue;
        result_frenet_paths.push_back(fp);
    }
}

// Return whether there is a collision in the given frenet_path
bool FrenetOptimalTrajectory::is_collision(FrenetPath& frenet_path) {
    // no obstacles
    if (obstacles.empty()) {
        return false;
    }

    // iterate over all obstacles
    for (tuple<double, double> obstacle : obstacles) {
        // calculate distance to each point in path
        for (int i = 0; i < frenet_path.x.size(); i++) {
            // exit if within OBSTACLE_RADIUS
            double xd = frenet_path.x[i] - get<0>(obstacle);
            double yd = frenet_path.y[i] - get<1>(obstacle);
            if (norm(xd, yd) <= OBSTACLE_RADIUS) {
                return true;
            }
        }
    }

    // no collisions
    return false;
}


