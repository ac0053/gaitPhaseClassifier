import numpy as np
import mujoco

def leg_height_con(theta, site_name, joint_id, desired_foot_positions, data, model):
    ceq = np.zeros([3, 1])

    data.qpos[joint_id] = theta
    mujoco.mj_forward(model, data)
    # calculate leg height based on height of leg from origin and spatial coordinates (x and z) of leg based on ThC
    # joint and tarsus tip
    ee_pos = data.site(site_name).xpos

    ceq = ee_pos - desired_foot_positions

    return np.array(ceq)