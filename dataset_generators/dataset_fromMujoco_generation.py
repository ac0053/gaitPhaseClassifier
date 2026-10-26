import csv
import datetime
import datetime
import json
import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from time import sleep
import mujoco.viewer as viewer
from matplotlib.path import Path as FilePath
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import pandas as pd
import math
from mujoco import mj_forward
from scipy.optimize import minimize    
from inverse_solvers.inverse_kinematics_nss import InverseKinematics
from inverse_solvers.inverse_dynamics import InverseDynamics
from inverse_solvers.leg_height_con import leg_height_con


"""
Script runs both the inverse kinematics and dynamics solvers, then saves 
joint angle data, joint velocities, and ground reaction force data for each leg into csv_trajectory_datasets folder.

Lastly, it generates feature data for the gait phase classifier neural network (main.py).
"""

"""setup values, load in the model, find end_effectors and joint names"""
# note: expects leg names to start with LH,LF,LM... leg_names = ["LH", "RH", "LM", "RM", "LF", "RF"]
"""setup values, load in the model, find end_effectors and joint names"""
# note: expects leg names to start with LH,LF,LM... leg_names = ["LH", "RH", "LM", "RM", "LF", "RF"]
swing_duration = 0.5  # s
stance_duty = 0.5 # 0.75#: 0.6
step_height = [.07, .07, .06, .06, .07, .07]
floor_level = 0.10  # 0.125  best pose:0.1
floor_name = "ground"
front_lat = 0.15
mid_lat = 0.3
hind_lat = 0.2
front_fore = 0.2
mid_fore = 0
hind_fore = -0.2
#  x is fore-aft; y is lateral; z is vertical.
desired_foot_positions = [[-hind_fore, -hind_lat, 0],
                          [-hind_fore, hind_lat, 0],
                          [-mid_fore, -mid_lat, 0],
                          [-mid_fore, mid_lat, 0],
                          [-front_fore, -front_lat, 0],
                          [-front_fore, front_lat, 0]]
spring = 0
max_joint_vel = 1.5
max_body_trans = 0.1 #0.085
max_body_rot = 0.2
i_matrix = []
contra_phase = 0.5
ipsi_phase = 0.5
max_stance_duration = 6
max_duty_cycle = max_stance_duration / (max_stance_duration + swing_duration)
end_effector_name = "_Tip"
thorax_name = "Thorax"  # Must match .xml

# if VS Code
# model = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),'xml',"Droso_float.xml"))
# if PyCharm
"""
model = mujoco.MjModel.from_xml_path(
    os.path.join(os.getcwd(), 'calculations/xml', "Drosophibot_floating_custom_actuators.xml"))
"""
mjspec = mujoco.MjSpec.from_file(os.path.join(
                                 os.getcwd(), 'xml', "Drosophibot_floating_custom_actuators.xml"))
model = mjspec.compile()

# load model and initial data of robot and graphs function
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


# find body and joint names, list end_effectors in a separate array
leg_names = ["LH", "RH", "LM", "RM", "LF", "RF"]
num_legs = 6
body_names = [model.body(i).name for i in range(model.nbody)]
site_names = [model.site(i).name for i in range(model.nsite)]
joint_names = [model.jnt(i).name for i in range(model.njnt)][1:] # removes the first joint which is the floating base
actuator_names = [model.actuator(i).name for i in range(model.nu)]
effectors = [None] * len(leg_names)
for sites in site_names:
    if end_effector_name in sites:
        if "LH" in sites:
            effectors[0] = sites
        elif "RH" in sites:
            effectors[1] = sites
        elif "LM" in sites:
            effectors[2] = sites
        elif "RM" in sites:
            effectors[3] = sites
        elif "LF" in sites:
            effectors[4] = sites
        else:
            effectors[5] = sites


"""section 1 - resting posture formulation"""

# find tibia length from model to use for posture scaling factor
body_id_r = data.body("RM_Tibia").id
body_id_l = data.body("LM_Tibia").id
geom_size = None
for geom_id in range(model.ngeom):
    if model.geom_bodyid[geom_id] == body_id_r or model.geom_bodyid[geom_id] == body_id_l:
        geom_size = model.geom_size[geom_id]
posture_scale_factor = geom_size[1] * 2

resting_posture = np.empty(6, dtype=object)
# for loop goes through each leg and optimizes to match data from actual insect (located in leg_height_con)
for i in range(num_legs):
    # find range of joint for leg from mujoco model and difference
    lb = []
    ub = []
    joint_id = []
    act_id = []
    for joints in joint_names:
        if joints[:2] == leg_names[i]:
            lb.append(model.jnt_range[model.joint(joints).id][0])
            ub.append(model.jnt_range[model.joint(joints).id][1])
            joint_id.append(model.joint(joints).id)
            
        
    joint_id = model.jnt_qposadr[joint_id]
    ub = np.array(ub)
    lb = np.array(lb)
    ran = ub - lb
    n = ran.size

    # find the mapping between x (the normalized variable), theta, the joint rotation. x = A * theta + b.
    a_inv = np.diag(ran)
    b = -lb / ran
    b = np.array(b)

    # initial state of mujoco model is halfway through the motion
    init_config = 0.5 + np.zeros(n)

    # minimize the scaling factor for each leg
    scale_fac = 1 / (n / 4)
    g_distance = lambda x: np.dot(x, scale_fac * np.eye(n).dot(x - np.ones(np.size(x))))
    bounds = [(0, 1)] * len(init_config)
    options = {"ftol": 1e-10, "maxiter": 1e3}
    # constrain the leg posture for the z (height)
    """
    if leg_names[i] in ["LF", "RF", "LM", "RM"]:
        C = np.eye(3)  # Control x, y, and z coordinate of the foot.
    else:
        C = np.array([[0, 1, 0], [0, 0, 1]])  # Only control y and z coordinate of the foot, leave x free.
    """
    C = np.eye(3)

    cons = ({'type': "eq", "fun": lambda x: np.dot(C, 
        leg_height_con(a_inv @ (x - b), effectors[i], joint_id, desired_foot_positions[i], data, model))})
    # {'type':"eq","fun":lambda x:np.dot(leg_height_con(a_inv@(x-b),site_names[i],leg_names[i],joint_id,floor_level[i],posture_scale_factor,terrain_flat,data,model,"Thorax"),[0,1,0])},
    # {'type':"eq","fun":lambda x:np.dot(leg_height_con(a_inv@(x-b),site_names[i],leg_names[i],joint_id,floor_level[i],posture_scale_factor,terrain_flat,data,model,"Thorax"),[0,0,1])})
    result = minimize(g_distance, init_config, method='SLSQP', bounds=bounds, constraints=cons, options=options)
    final_theta = result.x

    print("Leg " + leg_names[i])
    print(result.message)
    print("Constraint violation: " + str(np.dot(C, leg_height_con(a_inv @ (final_theta - b), effectors[i], joint_id,
                                                                  desired_foot_positions[i], data, model))))

    # denormalize the theta value
    rest_config = a_inv.dot(final_theta - b)

    # set resting posture to mujoco model
    model.qpos0[joint_id] = -rest_config
    print(data.qpos[0:6])
    data.qpos[joint_id] = 0
    resting_posture[i] = rest_config
    # mujoco.mj_forward(model, data)
    print("Resting config found: " + str(rest_config))

    print("Foot position: " + str(leg_height_con(np.zeros(np.size(rest_config)), effectors[i], joint_id, [0, 0, 0],
                                                 data, model)))
    print('\n')

    """
    plt.figure()
    plt.scatter(np.array(joint_id), rest_config)
    plt.scatter(np.array(joint_id), ub, marker="_")
    plt.scatter(np.array(joint_id), lb, marker="_")
    plt.draw()
    plt.show()
    """

# update model to show resting posture
# mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        # mujoco.mj_step(model, data)
        v.sync()

""" section 2 - inverse walking kinematics """

# p_base is a point between the hind legs that is used to track motion
foot_point_rest = data.site(effectors[0]).xpos
p_base = [foot_point_rest[0], 0, 0]

# these are values that create smooth motion in Mujoco and still solve the inverse kinematics

num_trans = 7
num_rot = 11
"""
num_trans = 5
num_rot = 7
"""
# initialize place to save each variable
samps_per_step = 40 + 1  # 161  # 41
full_kinematics = np.empty((6, num_trans, num_rot), dtype=object)
full_sw_kinematics = np.empty((6, num_trans, num_rot), dtype=object)
full_st_kinematics = np.empty((6, num_trans, num_rot), dtype=object)
pep_joint_angles = np.empty((6, num_trans, num_rot), dtype=object)
aep_joint_angles = np.empty((6, num_trans, num_rot), dtype=object)
foot_path_traj = np.empty((6, num_trans, num_rot), dtype=object)

ik = InverseKinematics(num_rot, num_trans, leg_names, effectors, joint_names, p_base,
                       stance_duty, max_body_trans, max_body_rot, swing_duration, step_height,
                       samps_per_step)

print("solving inverse kinematics")
for i in range(6):
    joint_id = []
    for joints in joint_names:
        if joints[:2] == leg_names[i]:
            joint_id.append(model.joint(joints).id)
    joint_id = model.jnt_qposadr[joint_id]
    print("joint_id for leg " + str(i) + ": " + str(joint_id))

    trans_mat, rot_mat, r_foot_wrt_base, leg_trans, leg_rot = ik.trans_rot_wrt_base(data, i, num_trans, num_rot)

    hh = plt.figure()
    gg = plt.figure()
    for j in range(num_trans):
        for k in range(num_rot):
            # Reset this leg to its resting posture before each independent IK solve.
            # This prevents state carry-over across (translation, rotation) trials.
            data.qpos[joint_id] = 0
            mujoco.mj_forward(model, data)

            stance_vec, j_p = ik.stance_vec_calc(r_foot_wrt_base, rot_mat[j, k], trans_mat[j, k])
            foot_path, foot_path_st_f_new, foot_path_sw_new_5th = ik.foot_trajectory(i, stance_vec)
            kinematics, step_frequency = ik.moore_penrose(model, data, i, r_foot_wrt_base, foot_path)

            kinematics_resample, st_kinematics, sw_kinematics, aep, pep, foot_path_resample = ik.kinematics_scale(
                kinematics, foot_path, foot_path_st_f_new, foot_path_sw_new_5th, step_frequency)
            full_kinematics[i, j, k] = kinematics_resample
            if i == 0 & j == 0 & k == 0:
                num_samples_per_step = np.shape(kinematics_resample)[1]

            full_sw_kinematics[i, j, k] = sw_kinematics
            full_st_kinematics[i, j, k] = st_kinematics
            pep_joint_angles[i, j, k] = pep
            aep_joint_angles[i, j, k] = aep
            foot_path_traj[i, j, k] = foot_path_resample

            """
            if j == (num_trans + 1) / 2 and k == (num_rot + 1) / 2:
            #if (i == 2) and (j > 1):
                print(str(i))
                print("centered!")
                plt.figure()
                for n in range(np.shape(kinematics)[0]):
                    plt.plot(np.tile(kinematics[n, :], (3, )), label="joint " + str(n))
                plt.legend()
                plt.grid()
                plt.title("leg " + str(i) + ", (j,k) = (" + str(j) + ", " + str(k) + ")")
                plt.ylabel("joint angles (rad)")
                plt.draw()

                plt.figure()
                for n in range(np.shape(foot_path)[0]):
                    plt.plot(np.tile(foot_path[n, :], (3, )), label="axis " + str(n))
                plt.legend()
                plt.grid()
                plt.title("leg " + str(i))
                plt.ylabel("foot path (m)")
                plt.draw()

                foot_path, foot_path_st_f_new, foot_path_sw_new_5th = ik.foot_trajectory(i, stance_vec, to_plot_raw=False)

                hh = plt.figure()
            """
        """
          plt.figure(hh)
             for n in range(np.shape(full_st_kinematics[i, 0, 0])[0]):
                plt.plot(full_st_kinematics[i, j, k][n, :])
            plt.title("leg " + str(i) + " stance phase angles")
            plt.ylabel("joint angles (rad)")
            plt.draw()

            plt.figure(gg)
            for n in range(np.shape(kinematics)[0]):
                plt.plot(kinematics[n, :])
            plt.title("leg " + str(i) + " joint angles UNSCALED")
            plt.ylabel("joint angles (rad)")
            plt.draw()"""

    # For each joint in this leg, plot the pep versus the body translation and rotation per step
"""
    jt_pep = np.empty((num_trans, num_rot))
    for j in range(len(joint_id)):
        for k in range(num_trans):
            for m in range(num_rot):
                jt_pep[k, m] = pep_joint_angles[i, k, m][j]


        # Plot the relationship between translation, rotation, and this joint's pep
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        surf = ax.plot_surface(trans_mat, rot_mat, jt_pep, cmap="coolwarm",
                               linewidth=0, antialiased=False)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)


print("inverse kinematics solved")
plt.show()
"""
# plot joint angles over time for each leg pair for a specific rotation and translation
# note: this is done before accounting for ipsa and contra phases so still pairs are still in tandem
# graphs.plot_inverse_kinematics(swing_duration,stance_duty,full_kinematics,0,0)


"""section 3 - shift gait"""

# change the walking gait to account for ipsalateral and contralateral phases to model bug walking
stance_ids = np.empty((num_legs, num_trans, num_rot), dtype=object)

for trans_idx in range(num_trans):
    for rot_idx in range(num_rot):
        phase_amount = np.zeros(num_legs, dtype=float)
        shift_amount = np.zeros(num_legs, dtype=int)

        for leg_idx in range(num_legs):
            traj_len = full_kinematics[leg_idx, trans_idx, rot_idx].shape[1]
            stance_len = full_st_kinematics[leg_idx, trans_idx, rot_idx].shape[1]
            half_len = math.floor(stance_len / 2)
            stance_id_pt_1 = np.arange(0, half_len, dtype=int)
            stance_id_pt_2 = np.arange(traj_len - half_len, traj_len, dtype=int)
            base_stance_ids = np.concatenate((stance_id_pt_1, stance_id_pt_2))

            leg_idx_matlab = leg_idx + 1
            if leg_idx_matlab == 1:
                phase_amount[leg_idx] = 0.0
            elif leg_idx_matlab % 2 == 0:
                phase_amount[leg_idx] = phase_amount[leg_idx - 1] + contra_phase
            else:
                phase_amount[leg_idx] = phase_amount[leg_idx - 2] + ipsi_phase

            # MATLAB wraps after 1; modulo keeps this robust for larger phase inputs.
            phase_amount[leg_idx] = phase_amount[leg_idx] % 1.0
            shift_amount[leg_idx] = int(round(phase_amount[leg_idx] * traj_len))
            stance_ids[leg_idx, trans_idx, rot_idx] = (base_stance_ids + shift_amount[leg_idx]) % traj_len

            # MATLAB shifts whenever phase > 0 (circshift by 0 is harmless if rounded to zero).
            if phase_amount[leg_idx] > 0.0:
                full_kinematics[leg_idx, trans_idx, rot_idx] = np.roll(full_kinematics[leg_idx, trans_idx, rot_idx],
                                                                       shift_amount[leg_idx], axis=1)
                foot_path_traj[leg_idx, trans_idx, rot_idx] = np.roll(foot_path_traj[leg_idx, trans_idx, rot_idx],
                                                                      shift_amount[leg_idx], axis=1)

""" section 5 - inverse dynamics"""
trans_vec = np.linspace(max_body_trans, -max_body_trans, num=num_trans)
rot_vec = np.linspace(max_body_rot, -max_body_rot, num=num_rot)
"""
# inverse dynamics calculations for joint torques and grf and offset of leg motion
id = InverseDynamics(model, data, full_kinematics, body_names, leg_names, joint_names, swing_duration, stance_duty,
                     contra_phase, ipsi_phase)

vel = [None] * 6
acc = [None] * 6
for i in range(6):
    vel[i], acc[i] = id.vel_acc(i, trans_vec, rot_vec)
tau, grf = id.solve_inverse_dynamics_with_contact_plane(trans_vec, rot_vec, foot_path_traj[:, 1, 1], vel=vel, acc=acc,
                                                        plane_geom_name="ground", )
tau_from_grf = id.estimate_torque_from_grf(trans_vec, rot_vec, grf)

traj = full_kinematics[0, 0, 0]
samps_per_step = id.samps_per_step
step_period = id.step_period
sample_idx = np.arange(samps_per_step)

# Shift plotted GRF so the selected leg stance phase is centered in time.
plot_leg = 0
stance_idx_leg = np.asarray(stance_ids[plot_leg, trans_vec, rot_vec], dtype=int)
center_idx = samps_per_step // 2
angles = 2.0 * np.pi * stance_idx_leg / samps_per_step
mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
if mean_angle < 0:
    mean_angle += 2.0 * np.pi
stance_center_idx = int(np.round((mean_angle / (2.0 * np.pi)) * samps_per_step)) % samps_per_step
plot_shift = center_idx - stance_center_idx
grf_plot = np.roll(grf[plot_leg, :, :], plot_shift, axis=0)

# Shift torque samples for the plotted leg so markers align with shifted GRF.
tau_plot = np.roll(tau[plot_leg, :], plot_shift, axis=0)
tau_from_grf_plot = np.roll(tau_from_grf[plot_leg, :], plot_shift, axis=0)

# Convert shifted stance indices into contiguous intervals for plotting.
stance_idx_shifted = (stance_idx_leg + plot_shift) % samps_per_step
stance_mask = np.zeros(samps_per_step, dtype=bool)
stance_mask[stance_idx_shifted] = True
transition_idx = np.diff(np.concatenate(([0], stance_mask.astype(int), [0])))
stance_starts = np.where(transition_idx == 1)[0]
stance_ends = np.where(transition_idx == -1)[0] - 1

time_vec = np.linspace(0, step_period, grf_plot.shape[0])
stance_intervals = [(time_vec[start], time_vec[end]) for start, end in zip(stance_starts, stance_ends)]

graphs.plot_grf(grf_plot, time_vec, plot_leg, stance_intervals=stance_intervals)
graphs.plot_torques(tau_plot, time_vec, plot_leg)
graphs.plot_torque_solution_method(tau_plot, tau_from_grf_plot, time_vec, plot_leg)
"""
""" section 6 - save data as CSV in respective folers"""

# data is grouped in folders - inverse kinematics, torques, and resting posture
folder_rest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resting_posture')
os.makedirs(folder_rest, exist_ok=True)
folder_kin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inverse_kinematics')
os.makedirs(folder_kin, exist_ok=True)
folder_torque = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'torques')
os.makedirs(folder_torque, exist_ok=True)

# resting posture
joint_id_rest = ["ThC1", "ThC3", "CTr", "TrF", "FTi"]
# input NaN values for top two joints not present in middle and hind legs for actuation of motors
# convert resting posture to 5x6 array
rest_post_5x6 = np.empty((5, 6), dtype=object)
for i in range(6):
    if i < 4:
        current = np.array(resting_posture[i])
        nan = np.empty(2)
        nan.fill(np.nan)
        rest_post_5x6[:, i] = np.concatenate((nan, current))
    else:
        rest_post_5x6[:, i] = resting_posture[i]

csv_path_rest = os.path.join(folder_rest, 'rest_angles.csv')
rest = pd.DataFrame(rest_post_5x6, index=joint_id_rest, columns=leg_names)
rest.to_csv(csv_path_rest, index=True, float_format='%.3f')

"""Map full_kinematics cell and grf to Mujoco model, then gathers joint kinematics and GRF data for one leg 
and their corresponding joints"""
"""
leg_idx legend:
0 = LH, 1 = RH, 2 = LM, 3 = RM, 4 = LF, 5 = RF
"""
joint_qpos_addrs = {}
joint_qvel_addrs = {}
effector_body_ids = {}
leg_joint_map = {}
leg_effector_map = {}
joint_id = []
act_id = []
dt = model.opt.timestep # 2 milliseconds
total_time = 5.0 # seconds
num_steps = int(round(total_time / dt))

for current_leg_name in leg_names:
    matched_joints = [j for j in joint_names if current_leg_name in j]
    leg_joint_map[current_leg_name] = matched_joints
    for j in matched_joints:
        joint_id = model.joint(j).id
        joint_qpos_addrs[j] = model.jnt_qposadr[joint_id]
        joint_qvel_addrs[j] = model.jnt_dofadr[joint_id]

csv_headers = ['timestamp']
for current_leg_name in leg_names:
    csv_headers.extend([f'{current_leg_name}_Tip_GRF_x', f'{current_leg_name}_Tip_GRF_y', f'{current_leg_name}_Tip_GRF_z'])
    site_name = f"{current_leg_name}_Tip"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in the model!")
    parent_body_id = model.site_bodyid[site_id]
    effector_body_ids[current_leg_name] = parent_body_id
    
    matched_joints = leg_joint_map[current_leg_name]
    for j in matched_joints:
        csv_headers.extend([f'{j}_cmd', f'{j}_angle', f'{j}_vel', f'{j}_acc'])
    mujoco.mj_step(model, data)

for trans_idx in range(num_trans):
    for rot_idx in range(num_rot):
        print(f"Running simulation for translation index {trans_idx} and rotation index {rot_idx}")
        root_body_id = model.body('Thorax').id 
        qpos_adr = model.body_jntadr[root_body_id] 

        # set position of Thorax body to start in middle of the floor and slightly above it to avoid initial penetration
        floor_center_x = 0.0
        floor_center_y = 0.0
        floor_clearance_z = 0.1   

        data.qpos[qpos_adr : qpos_adr + 3] = [floor_center_x, floor_center_y, floor_clearance_z]

        # run the simulation and collect data for the current translation and rotation indices
        with viewer.launch_passive(model, data) as v:
            print("press esc to exit view")
            row_data = []
            step = 0 # initialization for while loop
            
            while v.is_running() and step <= num_steps:
                current_time = step * dt
                row = [current_time]
        
                for leg_idx, current_leg_name in enumerate(leg_names):
                    matched_joints = leg_joint_map[current_leg_name]
                    kinematics_slice = full_kinematics[leg_idx, trans_idx, rot_idx][:, step % num_samples_per_step]
                    for idx, j in enumerate(matched_joints):
                        act_id = model.actuator(j).id
                        data.ctrl[act_id] = kinematics_slice[idx] # set the actuator control input to match the desired joint angle from the kinematics slice

                mujoco.mj_step(model, data)

                for leg_idx, current_leg_name in enumerate(leg_names):
                    total_grf_world = np.zeros(3) # initialize total ground reaction forces for the current leg's end effector
                    target_foot_id = effector_body_ids[current_leg_name] # get the body ID of the current leg's end effector
                    
                    for i in range(data.ncon):
                        contact = data.contact[i]
                        body1_id = model.geom_bodyid[contact.geom1]
                        body2_id = model.geom_bodyid[contact.geom2]

                        if body1_id == target_foot_id or body2_id == target_foot_id:
                            c_force = np.zeros(6) 
                            mujoco.mj_contactForce(model, data, i, c_force) # force of body1 on body2 in contact frame
            
                            contact_frame = contact.frame.reshape(3, 3)
                            force_world = contact_frame.T @ c_force[:3] # transform contact force to world frame
                            # if the target foot is body1, reverse the force direction to get the force on the foot (body2).
                            if body1_id == target_foot_id:
                                force_world = -force_world
                
                            total_grf_world += force_world
                            
                    row.append(total_grf_world[0])  # GRF x
                    row.append(total_grf_world[1])  # GRF y
                    row.append(total_grf_world[2])  # GRF z

                    matched_joints = leg_joint_map[current_leg_name]
                    kinematics_slice = full_kinematics[leg_idx, trans_idx, rot_idx][:, step % num_samples_per_step]
                    
                    for idx, j in enumerate(matched_joints):
                        row.append(kinematics_slice[idx])           # active joint command
                        row.append(data.qpos[joint_qpos_addrs[j]])  # joint angle
                        row.append(data.qvel[joint_qvel_addrs[j]])  # joint velocity
                        row.append(data.qacc[joint_qvel_addrs[j]])  # joint acceleration
    
                v.sync()
                sleep(0.01)  # adjust sleep time to control simulation speed
                row_data.append(row)
                step += 1
                
        df = pd.DataFrame(row_data, columns=csv_headers)
        df.to_csv(f"csv_trajectory_datasets/master_trajectories_tran{trans_idx}_rot{rot_idx}.csv", index=False)
        print(f"Dataset for tran{trans_idx}_rot{rot_idx} saved!")