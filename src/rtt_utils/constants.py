"""Table geometry (meters, world frame: table centre at the origin, z up) and reconstruction thresholds."""

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
TABLE_HEIGHT = 0.76  # z of the playing surface
NET_HEIGHT = 0.1525
NET_OVERHANG = 0.1525  # the net extends this far beyond each side of the table
TABLE_HALF_LENGTH = TABLE_LENGTH / 2
TABLE_HALF_WIDTH = TABLE_WIDTH / 2
NET_HALF_LENGTH = TABLE_HALF_WIDTH + NET_OVERHANG

# TT4D.determine_validity: a clip is rejected when the ODE fit RMSE of any segment, or the neural ball
# reprojection error normalized by the table diagonal, exceeds these.
MAX_RMSE_THRESHOLD = 0.3
MAX_NEURAL_REPROJECTION_THRESHOLD = 0.2
