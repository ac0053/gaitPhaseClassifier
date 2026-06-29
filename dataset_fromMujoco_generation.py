from matplotlib.path import Path as FilePath
import matplotlib.pyplot as plt
import os
import numpy as np
import mujoco
from mujoco import viewer
import time
import pandas as pd
import math
from mujoco import mj_forward
from mujoco import viewer
from scipy.optimize import minimize
from inverse_solvers import inverse_kinematics_interiorPenalty
from inverse_solvers.inverse_kinematics_interiorPenalty import InverseKinematics
from inverse_solvers.inverse_dynamics import InverseDynamics
import leg_height_con

"""
Script runs both the inverse kinematics and dynamics solvers, then maps values to
qpos and ctrl (torque) in Mujoco.

Lastly, it generates feature data for the gait phase classifier neural network.
"""

"""setup values, load in the model, find end_effectors and joint names"""
# note: expects leg names to start with LH,LF,LM... leg_names = ["LH", "RH", "LM", "RM", "LF", "RF"]
swing_duration=0.5 #s
stance_duty=0.6 #s
step_height=[.07,.07,.06,.06,.07,.07]
floor_level= 0.125
spring=0
max_joint_vel=1.5
max_body_trans=0.085
max_body_rot=0.2
i_matrix=[]
contra_phase=0.5
ipsal_phase=0.5
max_stance_duration = 6
max_duty_cycle = max_stance_duration / (max_stance_duration + swing_duration)
end_effector_name = "_Tip"


#load model and initial data
model = mujoco.MjModel.from_xml_path(os.path.join(os.getcwd(),'xml','Droso_float.xml'))
data = mujoco.MjData(model)

# find body and joint names, list end_effectors in a separate array
leg_names = ["LH", "RH", "LM", "RM", "LF", "RF"]
num_legs=6
body_names = [model.body(i).name for i in range(model.nbody)]
joint_names = [model.jnt(i).name for i in range(model.njnt)]
effectors=[None]*len(leg_names)
for bodies in body_names:
    if end_effector_name in bodies:
        if "LH" in bodies:
            effectors[0]=bodies
        elif "RH" in bodies:
            effectors[1]=bodies
        elif "LM" in bodies:
            effectors[2]=bodies
        elif "RM" in bodies:
            effectors[3]=bodies
        elif "LF" in bodies:
            effectors[4]=bodies
        else:
            effectors[5]=bodies


"""Resting Posture"""
"""section 1 - resting posture formulation"""
# find tibia length from model to use for posture scaling factor
body_id_r = data.body("RM_Tibia").id
body_id_l = data.body("LM_Tibia").id
geom_size = None
for geom_id in range(model.ngeom):
    if model.geom_bodyid[geom_id] == body_id_r or model.geom_bodyid[geom_id] == body_id_l:
        geom_size = model.geom_size[geom_id]
posture_scale_factor = geom_size[1] * 2

resting_posture = np.empty(6,dtype=object)
# for loop goes through each leg and optimizes to match data from actual insect (located in leg_height_con)
for i in range(num_legs):
    # find range of joint for leg from mujoco model and difference
    lb=[]
    ub=[]
    joint_id = []
    for joints in joint_names:
        if joints[:2] == leg_names[i]:
            lb.append(model.jnt_range[model.joint(joints).id][0])
            ub.append(model.jnt_range[model.joint(joints).id][1])
            joint_id.append(model.joint(joints).id)
    ub=np.array(ub)
    lb=np.array(lb)
    ran = ub - lb
    n = ran.size

    # find the mapping between x (the normalized variable), theta, the joint rotation. x = A * theta + b.
    a_inv = np.diag(ran)
    b = -lb/ran
    b=np.array(b)

    # initial state of mujoco model is halfway through the motion
    init_config=0.5+np.zeros(n)

    # minimize the scaling factor for each leg
    scale_fac = 1 / (n / 4)
    g_distance=lambda x:np.dot(x,scale_fac*np.eye(n).dot(x-np.ones(np.size(x))))
    bounds=[(0, 1)] * len(init_config)
    options={"ftol":1e-10}
    # constrain the leg posture for the z (height)
    cons=({'type':"eq","fun":lambda x:np.dot(leg_height_con.leg_height_con(a_inv@(x-b),effectors[i],leg_names[i],joint_id,floor_level,posture_scale_factor,data,model,"Thorax"),[1,0,0])})
            #{'type':"eq","fun":lambda x:np.dot(leg_height_con(a_inv@(x-b),site_names[i],leg_names[i],joint_id,floor_level[i],posture_scale_factor,terrain_flat,data,model,"Thorax"),[0,1,0])},
            #{'type':"eq","fun":lambda x:np.dot(leg_height_con(a_inv@(x-b),site_names[i],leg_names[i],joint_id,floor_level[i],posture_scale_factor,terrain_flat,data,model,"Thorax"),[0,0,1])})
    result = minimize(g_distance,init_config,method='SLSQP',bounds=bounds,constraints=cons,options=options)
    final_theta=result.x

    # denormalize the theta value
    rest_config=a_inv.dot(final_theta-b)

    # set resting posture to mujoco model
    data.qpos[joint_id] = rest_config
    resting_posture[i] = rest_config
    mujoco.mj_forward(model, data)
print('resting config found')


# update model to show resting posture
mujoco.mj_forward(model,data)

"""Inverse Kinematics Solver"""
foot_point_rest=data.body(effectors[0]).xpos
p_base=[foot_point_rest[0],0,floor_level]
#map IK solver to qpos data
num_trans=5
num_rot=7

# initialize place to save each variable
samps_per_step = 41
sample_idx = np.arange(samps_per_step)
full_kinematics = np.empty((6, num_trans,num_rot), dtype=object)
full_sw_kinematics = np.empty((6,num_trans,num_rot), dtype=object)
full_st_kinematics = np.empty((6, num_trans,num_rot), dtype=object)
pep_joint_angles=np.empty((6, num_trans,num_rot), dtype=object)
aep_joint_angles=np.empty((6, num_trans,num_rot), dtype=object)
foot_path_traj = np.empty((6, num_trans,num_rot), dtype=object)

print("solving inverse kinematics")
for i in range(6):
    joint_id = []
    for joints in joint_names:
        if joints[:2] == leg_names[i]:
            joint_id.append(model.joint(joints).id)
    for j in range(num_trans):
        for k in range(num_rot):
            # Reset this leg to its resting posture before each independent IK solve.
            # This prevents state carry-over across (translation, rotation) trials.
            data.qpos[joint_id] = resting_posture[i]
            mujoco.mj_forward(model, data)
            ik = InverseKinematics(model,data,num_rot,num_trans,leg_names,effectors,joint_names,p_base,stance_duty,max_body_trans,max_body_rot,swing_duration,step_height,samps_per_step)
            trans_mat,rot_mat,r_foot_wrt_base,leg_trans,leg_rot=ik.trans_rot_wrt_base(i,num_trans,num_rot)
            stance_vec,j_p = ik.stance_vec_calc(r_foot_wrt_base,rot_mat,trans_mat,i,j,k)
            foot_path,foot_path_st_f_new,foot_path_sw_new_5th=ik.foot_trajectory(i,stance_vec)
            kinematics,step_frequency = ik.moore_penrose(i,r_foot_wrt_base,foot_path)
            kinematics_resample,st_kinematics,sw_kinematics,aepid,pepid, foot_path_resample=ik.kinematics_scale(kinematics,foot_path,foot_path_st_f_new,foot_path_sw_new_5th,step_frequency)
            full_kinematics[i,j,k]=kinematics_resample
            full_sw_kinematics[i,j,k]=sw_kinematics
            full_st_kinematics[i,j,k]=st_kinematics
            pep_joint_angles[i,j,k]=kinematics_resample[:,pepid]
            aep_joint_angles[i,j,k]=kinematics_resample[:,aepid]
            foot_path_traj[i,j,k]=foot_path_resample
print("inverse kinematics solved")

"""Inverse Dynamics solver"""
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
                phase_amount[leg_idx] = phase_amount[leg_idx - 2] + ipsal_phase

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

"""Map full_kinematics cell and grf to Mujoco model, then gathers pos and vel data for one leg 
and their corresponding joints"""
all_joint_ids = []
all_joint_names = []
all_qpos_idx = []
all_dof_idx = []

leg_joint_qpos_indices = {leg: [] for leg in leg_names}
leg_joint_dof_indices = {leg: [] for leg in leg_names}

dt = model.opt.timestep

"""
leg_idx legend: 
0 = LH
1= RH
2 = LM
3 = RM
4 = LF
5 = RF
"""

leg_idx = 0
for j_name in joint_names:
    if leg_names[leg_idx] in j_name:
        j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        all_joint_ids.append(j_id)
        all_joint_names.append(j_name)
        all_qpos_idx.append(model.jnt_qposadr[j_id])
        all_dof_idx.append(model.jnt_dofadr[j_id])
        leg_joint_qpos_indices[(leg_names[leg_idx])].append(model.jnt_qposadr[j_id])
        leg_joint_dof_indices[(leg_names[leg_idx])].append(model.jnt_dofadr[j_id])

csv_headers = ["time"]

# make headers for csv. separates each joint into their pos values and grf for each leg (x, y, z) coords
for j in all_joint_names:
    print(j)
    csv_headers.append(f"{j}_pos")
    csv_headers.append(f"{j}_vel")
    csv_headers.append(f"{j}_acc")


"""TODO: add GRF data"""
"""for l in leg_names:
    csv_headers.append(f"{l}_GRF_x")
    csv_headers.append(f"{l}_GRF_y")
    csv_headers.append(f"{l}_GRF_z")
"""""

big_main_dataset = []
sequence_length = full_kinematics.shape[-1]

for step_count in range(10000):
    current_time = step_count * dt
    row_data = [current_time]
    data_idx = step_count % sequence_length #so the data can repeat itself
    #maps kinematics cell
    qpos_indices = leg_joint_qpos_indices
    dof_indices = leg_joint_dof_indices


    leg_name = leg_names[leg_idx]
    qpos_indices = leg_joint_qpos_indices[leg_name]
    dof_indices = leg_joint_dof_indices[leg_name]
    leg_positions = full_kinematics[leg_idx, 0, 0][:, data_idx]
    data.qpos[qpos_indices] = leg_positions

    mujoco.mj_step(model, data)

    for pos_idx, dof_idx in zip(all_qpos_idx, all_dof_idx):
        row_data.append(data.qpos[pos_idx])
        row_data.append(data.qvel[dof_idx])
        row_data.append(data.qacc[dof_idx])

    """#initialize contact forces for all legs to zero
    leg_grfs = {leg: np.zeros(3) for leg in leg_names}

    plane_geom_id = model.geom("ground").id
    tip_body_ids = [model.body(f"{name}_Tip").id for name in leg_names]

    for c_idx in range(data.ncon):
        contact = data.contact[c_idx]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, c_idx, wrench)

        for leg in leg_names:
            if (geom1_name and leg in geom1_name) or (geom2_name and leg in geom2_name):
                leg_grfs[leg] += wrench[:3]

    for leg in leg_names:
        row_data.extend(leg_grfs[leg])
"""

    big_main_dataset.append(row_data)
df = pd.DataFrame(big_main_dataset, columns=csv_headers)
df.to_csv(f"csv/gait_phase_{leg_name}_features.csv", index=False)

