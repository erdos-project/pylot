# Parameter
MAX_SPEED = 25.0            # maximum speed [m/s]
MAX_ACCEL = 6.0             # maximum acceleration [m/ss]
MAX_CURVATURE = 40          # maximum curvature [1/m]
MAX_ROAD_WIDTH = 8.0        # maximum road width [m]
D_ROAD_W = 1                # road width sampling length [m]
DT = 0.25                   # time tick [s]
MAXT = 6                    # max prediction time [m]
MINT = 4                    # min prediction time [m]
D_T_S = 1.0                 # target speed sampling length [m/s]
N_S_SAMPLE = 1              # sampling number of target speed
OBSTACLE_RADIUS = 3.5       # obstacle radius [m]

# cost weights
KJ = 0.1                    # jerk cost
KT = 0.1                    # time cost
KD = 1.0                    # end state cost
KLAT = 1.0                  # lateral cost
KLON = 1.0                  # longitudinal cost