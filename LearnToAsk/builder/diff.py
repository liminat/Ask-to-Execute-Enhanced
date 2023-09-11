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
    # check if there is an existing block at block's location
    if extra_check and conflicting_block_exists(block, built_config):
        return False

    # check if block is on ground
    if block_on_ground(block):
        return True

    # check if block has a supporting block
    if block_with_support(block, built_config):
        return True
    else:
        return False

def conflicting_block_exists(block, built_config):
    for existing_block in built_config:
        if conflicts(existing_block, block):
            return True

    return False

def conflicts(existing_block, block):
    return existing_block["x"] == block["x"] and existing_block["y"] == block["y"] and existing_block["z"] == block["z"]

def block_on_ground(block):
    return block["y"] == 1

def block_with_support(block, built_config):
    for existing_block in built_config:
        if supports(existing_block, block):
            return True

    return False

def supports(existing_block, block):
    x_support = abs(existing_block["x"] - block["x"]) == 1 and existing_block["y"] == block["y"] and existing_block["z"] == block["z"]

    y_support = abs(existing_block["y"] - block["y"]) == 1 and existing_block["x"] == block["x"] and existing_block["z"] == block["z"]

    z_support = abs(existing_block["z"] - block["z"]) == 1 and existing_block["x"] == block["x"] and existing_block["y"] == block["y"]

    return x_support or y_support or z_support

def get_diff(gold_config, built_config, optimal_alignment=None):
    """
    Args:
        gold_config: Gold configuration
        built_config: Configuration built so far
        Both are lists of dicts. Each dict contains info on block type and block coordinates.

    Returns:
        A minimal diff in built config space -- in terms of placement and removal actions --
        to take the built config state to the goal config state

        All minimal diffs (each in both built and gold config space) with corresponding complementary info --
        complementary info would be the original built config, a perturbed config and the transformation to transform
        the former into the latter
    """

    # generate all possible perturbations of built config in the build region
    perturbations = generate_perturbations(built_config, gold_config = gold_config, optimal_alignment = optimal_alignment)

    # compute diffs for each perturbation
    diffs = list([diff(go