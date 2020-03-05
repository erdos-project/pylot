import fot_wrapper
import time
import numpy as np
import matplotlib.pyplot as plt


def main():
    """A debug script for the frenet optimal trajectory planner.

    This script will solve the frenet optimal trajectory problem in a
    standalone simulation and visualize the results or raise an error if a
    path is not found.

    To run, replace `conds = {}` with the corresponding output from the pylot
    debug logs. The output will be from the `fot_planning_operator` and it will
    look like `initial_conditions = {...}`.
    """
    print(__file__ + " start!!")
    sim_loop = 40
    area = 40.0  # animation area length [m]
    show_animation = True

    conds = {'s0': 12.59999999999999,
             'c_speed': 7.10126796146418,
             'c_d': 0.10761995326139745,
             'c_d_d': 0.009671559894548877,
             'c_d_dd': 0.0,
             'wx': [132.67, 128.67, 124.67, 120.67, 116.67, 112.67, 108.67,
                    104.67, 101.43,  97.77,  94.84,  92.89,  92.4 ,  92.4 ,
                    92.4 ,  92.4 ,  92.4 ,  92.4 ,  92.4 ,  92.39,  92.39,
                    92.39,  92.39,  92.39,  92.39],
             'wy': [195.14, 195.14, 195.14, 195.14, 195.14, 195.14, 195.14,
                    195.14, 195.14, 195.03, 193.88, 191.75, 188.72, 185.32,
                    181.32, 177.32, 173.32, 169.32, 165.32, 161.32, 157.32,
                    153.32, 149.32, 145.32, 141.84],
             'obstacle_list': [[92.89, 191.75]],
             'x': 120.06613159179688,
             'y': 195.03477478027344,
             'vx': -7.101262092590332,
             'vy': 0.009129776619374752}  # paste output from debug log

    # way points
    wx = np.array(conds['wx'])
    wy = np.array(conds['wy'])

    # initial conditions
    x = conds['x']
    y = conds['y']

    # obstacle lists
    obs = np.array(conds['obstacle_list'])

    # initial state
    c_speed = conds['c_speed']  # current speed [m/s]
    c_d = conds['c_d']  # current lateral position [m]
    c_d_d = conds['c_d_d']  # current lateral speed [m/s]
    c_d_dd = conds['c_d_dd']  # current latral acceleration [m/s]
    s0 = conds['s0']  # current course position

    for i in range(sim_loop):
        print("Iteration: {}".format(i))
        start_time = time.time()
        result_x, result_y, _, params, success = \
            fot_wrapper.get_fot_frenet_space(s0, c_speed, c_d, c_d_d, c_d_dd,
                                             wx, wy, obs, 10)
        end_time = time.time() - start_time
        print("Time taken: {}".format(end_time))
        s0 = params['s0']
        c_d = params['c_d']
        c_d_d = params['c_d_d']
        c_d_dd = params['c_d_dd']
        c_speed = params['c_speed']
        if np.hypot(result_x[1] - wx[-1], result_y[1] - wy[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None]
            )
            plt.plot(wx, wy)
            if obs.shape[0] == 0:
                obs = np.empty((0, 2))
            plt.scatter(obs[:, 0], obs[:, 1], marker='o', s=(3*6)**2)
            plt.plot(result_x[1:], result_y[1:], "-or")
            plt.plot(result_x[1], result_y[1], "vc")
            plt.xlim(result_x[1] - area, result_x[1] + area)
            plt.ylim(result_y[1] - area, result_y[1] + area)
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.title("v[m/s]:" + str(c_speed)[0:4])
            plt.grid(True)
            plt.pause(0.1)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.1)
        plt.show()


if __name__ == '__main__':
    main()
