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
             'obstacle_list': [[118.2, 198.96], [118.78, 198.94],
                               [119.36, 198.91], [119.94, 198.88],
                               [120.52, 198.86], [121.1, 198.83],
                               [121.68, 198.81], [122.26, 198.78],
                               [122.85, 198.76], [123.43, 198.73]],
             'x': 120.06613159179688,
             'y': 195.03477478027344,
             'vx': -7.101262092590332,
             'vy': 0.009129776619374752}  # paste output from debug log

    # way points
    wx = conds['wx']
    wy = conds['wy']

    # initial conditions
    x = conds['x']
    y = conds['y']

    # obstacle lists
    ob = np.array(conds['obstacle_list'])

    # plot the environment
    plt.figure(figsize=(16, 16))
    plt.scatter(ob[:, 0], ob[:, 1], c='r', marker='x', label="Obstacles")
    plt.scatter(wx, wy, c='g', marker='.', label="Global Waypoints")
    plt.scatter(x, y, c='b', marker='o', label="Ego location")
    plt.xlim(x - area, x + area)
    plt.ylim(y - area, y + area)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.title("Initial environment")
    plt.figtext(0.2, 0.05,
                "Note: These are CARLA world coordinates and not true x, y"
                " space values. The resulting plots may be 'flipped'.",
                weight='extra bold',
                size='large')
    plt.legend()
    plt.show()

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = conds['c_speed']  # current speed [m/s]
    c_d = conds['c_d']  # current lateral position [m]
    c_d_d = conds['c_d_d']  # current lateral speed [m/s]
    c_d_dd = conds['c_d_dd']  # current latral acceleration [m/s]
    s0 = conds['s0']  # current course position


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
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
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
