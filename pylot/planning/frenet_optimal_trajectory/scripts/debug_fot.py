import matplotlib.pyplot as plt
import numpy as np

from pylot.planning.frenet_optimal_trajectory.frenet_optimal_trajectory \
    import generate_target_course, frenet_optimal_planning


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
    sim_loop = 500
    show_animation = True

    conds = {}  # paste output from debug log

    # way points
    wx = conds['wx']
    wy = conds['wy']

    # obstacle lists
    ob = np.array(conds['obstacle_list'])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = conds['c_speed']  # current speed [m/s]
    c_d = conds['c_d']  # current lateral position [m]
    c_d_d = conds['c_d_d']  # current lateral speed [m/s]
    c_d_dd = conds['c_d_dd']  # current latral acceleration [m/s]
    s0 = conds['s0']  # current course position

    area = 20.0  # animation area length [m]

    for i in range(sim_loop):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob, 10)
        assert path is not None, "Optimal trajectory not found."

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None]
            )
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[m/s]:" + str(c_speed)[0:4])
            plt.grid(True)
            plt.pause(0.001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
