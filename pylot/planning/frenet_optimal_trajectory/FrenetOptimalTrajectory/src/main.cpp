#include "FrenetOptimalTrajectory.h"
#include "FrenetPath.h"

#include <iostream>
#include <ctime>
#include <vector>
#include <tuple>

using namespace std;

int main() {
    // set up experiment
    clock_t start;
    double duration;
    int sim_loop = 40;
    double s0 = 12.6;
    double c_speed = 7.1;
    double c_d = 0.1;
    double c_d_d = 0.01;
    double c_d_dd = 0.0;
    vector<double> wx = {132.67, 128.67, 124.67, 120.67, 116.67, 112.67, 108.67,
                         104.67, 101.43,  97.77,  94.84,  92.89,  92.4 ,  92.4 ,
                         92.4 ,  92.4 ,  92.4 ,  92.4 ,  92.4 ,  92.39,  92.39,
                         92.39,  92.39,  92.39,  92.39};
    vector<double> wy = {195.14, 195.14, 195.14, 195.14, 195.14, 195.14, 195.14,
                         195.14, 195.14, 195.03, 193.88, 191.75, 188.72, 185.32,
                         181.32, 177.32, 173.32, 169.32, 165.32, 161.32, 157.32,
                         153.32, 149.32, 145.32, 141.84};
    vector<tuple<double, double>> obstacles;
    vector<double> ox = {98.2, 98.78, 99.36, 99.94, 100.52, 101.1, 101.68, 102.26, 102.85, 103.43};
    vector<double> oy = {198.96, 198.94, 198.91, 198.88, 198.86, 198.83, 198.81, 198.78, 198.76, 198.73};
    for (int i = 0; i < ox.size(); i++) {
        tuple<double, double> ob (ox[i], oy[i]);
        obstacles.push_back(ob);
    }

    // run experiment
    start = clock();
    for (int i = 0; i < 40; i++) {
        FrenetOptimalTrajectory fot = FrenetOptimalTrajectory(wx, wy, s0, c_speed, c_d, c_d_d, c_d_dd, 10, obstacles);
        FrenetPath* best_frenet_path = fot.getBestPath();
        s0 = best_frenet_path->s[1];
        c_d = best_frenet_path->d[1];
        c_d_d = best_frenet_path->d_d[1];
        c_d_dd = best_frenet_path->d_dd[1];
        c_speed = best_frenet_path->s_d[1];
    }
    duration = (clock() - start) / (double) CLOCKS_PER_SEC;
    cout << "Total time taken: " << duration << endl;
    cout << "Time per iteration: " << duration / sim_loop << endl;
    return 1;
}
