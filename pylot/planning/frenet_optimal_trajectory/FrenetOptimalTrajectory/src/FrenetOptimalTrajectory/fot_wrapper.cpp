#include "FrenetOptimalTrajectory.h"
#include "FrenetPath.h"
#include "CubicSpline2D.h"
#include "utils.h"

#include <iostream>
#include <vector>

using namespace std;

// C++ wrapper to expose the FrenetOptimalTrajectory class to python
extern "C" {
    // Compute the frenet optimal trajectory given initial conditions
    // in frenet space.
    //
    // Arguments:
    //      s0: initial longitudinal position along spline
    //      c_speed: initial velocity
    //      c_d: initial lateral offset
    //      c_d_d: initial lateral velocity
    //      c_d_dd: initial lateral acceleration
    //      xp, yp: list of waypoint coordinates
    //      xo, yo: list of obstacle coordinates
    //      np, no: length of the waypoint list and obstacle list
    //      target_speed: target velocity in [m/s]
    // Returns:
    //      x_path, y_path: the frenet optimal trajectory in cartesian space
    //      misc: the next states s0, c_speed, c_d, c_d_d, c_d_dd
    int get_fot_frenet_space(
            double s0, double c_speed, double c_d, double c_d_d, double c_d_dd,
            double* xp, double* yp, double* xo, double* yo, int np, int no,
            double target_speed, double* x_path, double* y_path,
            double* speeds, double* misc
            ) {
        vector<double> wx (xp, xp + np);
        vector<double> wy (yp, yp + np);

        vector<tuple<double, double>> obstacles;
        vector<double> ox (xo, xo + no);
        vector<double> oy (yo, yo + no);
        for (int i = 0; i < ox.size(); i++) {
            tuple<double, double> ob (ox[i], oy[i]);
            obstacles.push_back(ob);
        }

        FrenetOptimalTrajectory fot = FrenetOptimalTrajectory(wx, wy, s0,
                c_speed, c_d, c_d_d, c_d_dd, target_speed, obstacles);
        FrenetPath* best_frenet_path = fot.getBestPath();

        int success = 0;
        if (!best_frenet_path->x.empty()){
            int last = 0;
            for (int i = 0; i < best_frenet_path->x.size(); i++) {
                x_path[i] = best_frenet_path->x[i];
                y_path[i] = best_frenet_path->y[i];
                speeds[i] = best_frenet_path->s_d[i];
                last += 1;
            }

            // indicate last point in the path
            x_path[last] = NAN;
            y_path[last] = NAN;
            speeds[last] = NAN;

            misc[0] = best_frenet_path->s[1];
            misc[1] = best_frenet_path->s_d[1];
            misc[2] = best_frenet_path->d[1];
            misc[3] = best_frenet_path->d_d[1];
            misc[4] = best_frenet_path->d_dd[1];
            success = 1;
        }
        return success;
    }

    // Convert the initial conditions from cartesian space to frenet space
    void compute_initial_conditions(
            double s0, double x, double y, double vx,
            double vy, double forward_speed, double* xp, double* yp, int np,
            double* initial_conditions
            ) {
        vector<double> wx (xp, xp + np);
        vector<double> wy (yp, yp + np);
        CubicSpline2D* csp = new CubicSpline2D(wx, wy);

        // get distance from car to spline and projection
        double s = csp->find_s(x, y, s0);
        double distance = norm(csp->calc_x(s) - x, csp->calc_y(s) - y);
        tuple<double, double> bvec ((csp->calc_x(s) - x) / distance,
                (csp->calc_y(s) - y) / distance);

        // normal spline vector
        double x0 = csp->calc_x(s0);
        double y0 = csp->calc_y(s0);
        double x1 = csp->calc_x(s0 + 2);
        double y1 = csp->calc_y(s0 + 2);

        // unit vector orthog. to spline
        tuple<double, double> tvec (y1-y0, -(x1-x0));
        as_unit_vector(tvec);

        // compute tangent / normal car vectors
        tuple<double, double> fvec (vx, vy);
        as_unit_vector(fvec);

        // get initial conditions in frenet frame
        initial_conditions[0] = s; // current longitudinal position s
        initial_conditions[1] = forward_speed; // speed [m/s]
        // lateral position c_d [m]
        initial_conditions[2] = copysign(distance, dot(tvec, bvec));
        // lateral speed c_d_d [m/s]
        initial_conditions[3] = forward_speed * dot(tvec, fvec);
        initial_conditions[4] = 0.0; // lateral acceleration c_d_dd [m/s^2]
        // TODO: add lateral acceleration when CARLA 9.7 is patched (IMU)

        delete csp;
    }
}
