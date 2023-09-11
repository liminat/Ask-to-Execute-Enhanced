import numpy as np, sys, random
from scipy.spatial import distance

# NOTE: THIS CODE NEEDS TO BE MAINTAINED FOR BOTH PYTHON 2 AND 3 COMPATIBILITY

build_region_specs = {
    "x_min_build": -5,
    "x_max_build": 5,
    "y_min_build": 1,
    "y_max_build": 9,
    "z_min_build": -5,
    "z_max_build": 5
}

all_possible_rot_values = [0, 90, 180, -90, -180]
rot_matrices_dict = {}
for rot_value in all_possible_rot_values:
    theta_yaw = np.radians(-1 * rot_value)
    c, s = np.cos(theta_yaw), np.sin(theta_yaw)
    R_yaw = np.matrix([ [ c, 0, -s ], [ 0, 1, 0 ], [ s, 0, c ] ])
    rot_matrices_dict[rot_value] = R_yaw

def is_feasible_next_placement(block, built_config, extra_check):
    # check if there is an existing block at bl