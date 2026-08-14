import numpy as np
import mujoco
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
import copy


class InverseKinematics:
    def __init__(self, model, data, num_rot, num_trans, leg_names, effectors, joint_names, p_base, stance_duty,
                 max_body_trans, max_body_rot, swing_duration, step_height, samps_per_step):
        self.model = model
        self.data = data
        self.leg_names = leg_names
        self.effectors = effectors
        self.joint_names = joint_names
        self.p_base = p_base
        self.stance_duty = stance_duty
        self.max_body_trans = max_body_trans
        self.max_body_rot = max_body_rot
        self.swing_duration = swing_duration
        self.step_height = step_height
        self.num_rot = num_rot
        self.num_trans = num_trans
        self.num_samps_per_step = samps_per_step

        self.num_trials = num_rot * num_trans

        self.count = 0
        self.wrong = 0
        if self.wrong > 0:
            print("This rot and trans results in signifcant error")
        if self.count > 0:
            print("The path for this rot and mat could not be found")

    # specify rotations and translations wrt base point
    def trans_rot_wrt_base(self, leg, num_trans, num_rot):
        leg_trans = np.linspace(-self.max_body_trans, self.max_body_trans, num_trans)
        leg_rot = np.linspace(-self.max_body_rot, self.max_body_rot, num_rot)
        trans_mat = np.meshgrid(leg_trans, leg_rot)[0]

        rot_mat = np.zeros((num_trans, num_rot))
        for i in range(num_trans):
            if leg_trans[i] < 0:
                theta_lim = leg_trans[i] * self.max_body_rot / self.max_body_trans + self.max_body_rot
            else:
                theta_lim = -leg_trans[i] * self.max_body_rot / self.max_body_trans + self.max_body_rot
            rot_mat[i, :] = np.linspace(-theta_lim, theta_lim, num_rot)
        rot_mat = rot_mat.T

        # find position of each foot wrt base point- will be used to find the foot velocity for each trans and rot velocity
        p_foot = self.data.body(self.effectors[leg]).xpos
        r_foot_wrt_base = p_foot - self.p_base

        return trans_mat, rot_mat, r_foot_wrt_base, leg_trans, leg_rot

    # stance_vec is the position of the foot of a certain leg at a specific rotation and translation
    def stance_vec_calc(self, r_foot_wrt_base, rot_mat, trans_mat, leg, tran, rot):
        j_p = np.array(r_foot_wrt_base)
        psi = rot_mat[rot, tran]
        r = [trans_mat[rot, tran], 0, 0]
        r_one_frame = np.array(
            [[math.cos(psi / (self.num_samps_per_step - 1)), -math.sin(psi / (self.num_samps_per_step - 1)), 0],
             [math.sin(psi / (self.num_samps_per_step - 1)), math.cos(psi / (self.num_samps_per_step - 1)), 0],
             [0, 0, 1], ])
        r_one_frame_neg = np.array(
            [[math.cos(-psi / (self.num_samps_per_step - 1)), -math.sin(-psi / (self.num_samps_per_step - 1)), 0],
             [math.sin(-psi / (self.num_samps_per_step - 1)), math.cos(-psi / (self.num_samps_per_step - 1)), 0],
             [0, 0, 1]])
        r_one_frame_r = [x / (self.num_samps_per_step - 1) for x in r]

        mid = (self.num_samps_per_step + 1) // 2
        # stance_vec is the position of the foot
        stance_vec = np.zeros((3, self.num_samps_per_step))
        stance_vec[:, mid - 1] = j_p

        for m in range(mid, self.num_samps_per_step):
            stance_vec[:, m] = r_one_frame_neg @ stance_vec[:, m - 1] - r_one_frame_r
        for m in range(mid - 2, -1, -1):
            stance_vec[:, m] = r_one_frame @ (stance_vec[:, m + 1] + r_one_frame_r)
        for v in range(len(stance_vec)):
            stance_vec[v] = stance_vec[v] - j_p[v]

        return stance_vec, j_p

    def foot_trajectory(self, leg, stance_vec, to_plot_raw=False):
        # generate swing profile with 5th order polynomials
        q5 = lambda t, p: p[0] + p[1] * t + p[2] * t ** 2 + p[3] * t ** 3 + p[4] * t ** 4 + p[5] * t ** 5

        step_num = stance_vec.shape[1]
        x_max = stance_vec[0][-1]
        x_min = stance_vec[0][0]
        t0 = 0
        tf = self.swing_duration
        t_mid = tf / 2

        a5_13 = np.array(
            [[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
             [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
             [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
             [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
             [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
             [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])
        a5_12 = np.array(
            [[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
             [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
             [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3],
             [1, t_mid, t_mid ** 2, t_mid ** 3, t_mid ** 4, t_mid ** 5],
             [0, 1, 2 * t_mid, 3 * t_mid ** 2, 4 * t_mid ** 3, 5 * t_mid ** 4],
             [0, 0, 2, 6 * t_mid, 12 * t_mid ** 2, 20 * t_mid ** 3]])
        a5_23 = np.array(
            [[1, t_mid, t_mid ** 2, t_mid ** 3, t_mid ** 4, t_mid ** 5],
             [0, 1, 2 * t_mid, 3 * t_mid ** 2, 4 * t_mid ** 3, 5 * t_mid ** 4],
             [0, 0, 0, 6 * t_mid, 12 * t_mid ** 2, 20 * t_mid ** 3],
             [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
             [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
             [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])

        stance_duration = self.swing_duration / (1 - self.stance_duty) * self.stance_duty
        swing_x_vel = (x_min - x_max) / stance_duration
        swing_x_acc = 0
        single_time_step = (self.swing_duration / (1 - self.stance_duty) * self.stance_duty) / step_num
        swing_z_vel1 = (stance_vec[2, -1] - stance_vec[2, -2]) / single_time_step
        swing_z_vel2 = (stance_vec[2, 1] - stance_vec[2, 0]) / single_time_step

        b5_12z = np.array([[stance_vec[2, -1], swing_z_vel1, 0, self.step_height[leg], 0, 0]]).T
        b5_23z = np.array([self.step_height[leg], 0, 0, stance_vec[2, 0], swing_z_vel2, 0]).T
        b5_13x = np.array([x_max, -swing_x_vel, swing_x_acc, x_min, -swing_x_vel, swing_x_acc]).T

        t = np.linspace(t0, tf, step_num)
        t1 = t[range(0, round((step_num - 1) / 2))]
        t2 = t[round(step_num / 2):]

        p5_1z = np.linalg.solve(a5_12, b5_12z)
        p5_2z = np.linalg.solve(a5_23, b5_23z)
        p5_x = np.linalg.solve(a5_13, b5_13x)

        z5_1 = q5(t1, p5_1z)
        z5_2 = q5(t2, p5_2z)
        z5_sw = np.append(z5_1, z5_2)
        x5_sw = q5(t, p5_x)

        row = stance_vec[0, :]
        diffs = np.diff(row)
        increasing = np.all(diffs >= 0)
        decreasing = np.all(diffs <= 0)
        invalid = False
        if not increasing and not decreasing:
            print(stance_vec[0, :])
            self.count += 1
            invalid = True

        # interpolate z coordinates in swing based stance
        if invalid:
            y5_sw = np.zeros((1, len(x5_sw)))
        elif np.linalg.norm(x5_sw) != 0:
            # Pchip only allows for strictly increasing monotonic series, need to check if strictly decreasing
            check = True
            for b in range(len(stance_vec[0, :]) - 1):
                if stance_vec[0, b] > stance_vec[0, b + 1]:
                    check = False

            if check:
                interp = interpolate.PchipInterpolator(stance_vec[0, :], stance_vec[1, :], extrapolate=True)
                y5_sw = interp(x5_sw)
            else:
                stance_x_reverse = []
                stance_z_reverse = []
                for b in range(len(stance_vec[0, :]) - 1, -1, -1):
                    stance_x_reverse.append(stance_vec[0, b])
                    stance_z_reverse.append(stance_vec[1, b])
                interp = interpolate.PchipInterpolator(stance_x_reverse, stance_z_reverse, extrapolate=True)
                y5_sw_reverse = interp(x5_sw)
                y5_sw = []
                for b in range(y5_sw_reverse.shape[0] - 1, -1, -1):
                    y5_sw.append(y5_sw_reverse[b])
        else:
            y5_sw = np.zeros((1, len(x5_sw)))
        y5_sw = np.array(y5_sw)
        foot_path_sw_new_5th = np.zeros((3, self.num_samps_per_step))
        foot_path_sw_new_5th[0, :] = x5_sw
        foot_path_sw_new_5th[1, :] = y5_sw[::-1]
        foot_path_sw_new_5th[2, :] = z5_sw

        foot_path_st_f_new = stance_vec[:, int(np.ceil(stance_vec.shape[1] / 2 - 1)):]
        foot_path_st_b_new = stance_vec[:, :int(np.floor(stance_vec.shape[1] / 2))]

        end_stf = foot_path_st_f_new.shape[1]
        end_sw = foot_path_sw_new_5th.shape[1]
        index = end_stf - 1

        foot_path = np.concatenate([foot_path_st_f_new[:, :-1],
                                    foot_path_sw_new_5th[:, :-1],
                                    foot_path_st_b_new], axis=1)

        #  foot_path is perfect at this point, as indicated by the plot:
        if to_plot_raw:
            plt.figure()
            for n in range(np.shape(foot_path)[0]):
                plt.plot(foot_path[n, :], label="axis " + str(n))
            plt.legend()
            plt.title("perfect leg")
            plt.ylabel("foot path (m)")
            plt.draw()
            plt.show()

        return foot_path, foot_path_st_f_new, foot_path_sw_new_5th

    # at a specific translation and rotation, use moore_penorse to solve for theta
    def moore_penrose(self, leg, r_foot_wrt_base, foot_path):
        # find active joints for the leg
        joint_id = []
        for joints in self.joint_names:
            if self.leg_names[leg] in joints:
                joint_id.append(self.model.joint(joints).id)

        # if leg has a body joint then flex it proportionally to the heading angle
        p_foot = self.data.body(self.effectors[leg]).xpos
        if any(np.isnan(p_foot)):
            # This leg does not have a resting posture
            raise Exception("Leg with name \"" + self.leg_names[leg] + "\" has no rest posture defined, "
                            "so inverse kinematics cannot be calculated.")

        r_foot_wrt_base[:] = p_foot - self.p_base

        stance_duration = self.swing_duration / (1 - self.stance_duty) * self.stance_duty
        step_frequency = 1 / (stance_duration + self.swing_duration)

        # save resting posture for easier use to solve for aep and pep configurations
        resting = self.data.qpos[joint_id]

        # figure out what IDs AEP and PEP are for this foot trajectory
        # use moore-penrose pseudo-inverse of the body-fram jacobian. Will find the joint velocities that move the foot
        # along the desired trajectories with the least joint motion possible
        # save position of the root body for comparison to foot positions
        q_vec = []
        for joints in joint_id:
            q_vec.append(self.data.qpos[joints])
        body_center = self.data.body("Thorax").xpos - self.p_base
        end_effector = self.data.body(self.effectors[leg]).xpos
        start_point = end_effector - self.p_base

        num_steps = foot_path.shape[1]
        kinematics = np.empty((len(joint_id), num_steps))
        r_foot = np.empty((3, num_steps))
        r_foot[:, 0] = start_point
        q_vec = np.array(q_vec)
        kinematics[:, 0] = q_vec.T
        orig_joints = joint_id

        # compute joint kinematics for the desired motion
        ind = 1
        to_continue = True
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        body_id = self.model.body(self.effectors[leg]).id
        lower_limit = self.model.jnt_range[joint_id, 0]
        upper_limit = self.model.jnt_range[joint_id, 1]

        while ind < num_steps and to_continue:
            # Plug in previous joint angles (kinematics) and advance the model.
            self.data.qpos[orig_joints] = kinematics[:, ind - 1]
            mujoco.mj_forward(self.model, self.data)

            # Foot position at previous joint angles
            effector_position = self.data.body(body_id).xpos
            r_foot[:, ind] = effector_position - self.p_base  # Not sure this is needed here.

            # Get the jacobian of this leg's end effector when it is in this configuration.
            # jacp is the position-specific component, jacr is the rotation-specific component. Do we need jacr here?
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            jac_active_init = jacp[:, joint_id]

            #  FOR MIDPOINT METHOD, step the model forward a half step:
            dx = foot_path[:, ind] - foot_path[:, ind - 1]
            try:
                self.data.qpos[orig_joints] = kinematics[:, ind - 1] + 0.5 * np.linalg.pinv(jac_active_init) @ dx
            except np.linalg.LinAlgError:
                1 + 1

            mujoco.mj_forward(self.model, self.data)

            # Get the jacobian of this leg's end effector in the "half-step" configuration.
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            jac_active_half = jacp[:, joint_id]
            # Half-step complete
            #
            # now the full step, using the half-step Jacobian and the full step distance.
            #
            kinematics[:, ind] = kinematics[:, ind - 1] + np.linalg.pinv(jac_active_half) @ dx
            ind += 1

        return kinematics, step_frequency

    def kinematics_scale(self, kinematics, foot_path, foot_path_st_f_new, foot_path_sw_new_5th, step_frequency):
        # scale the kinematics to the desired frequency and duty
        num_samps = kinematics.shape[1]
        # indices of first stance and swing phases
        # st1 = np.linspace(0, foot_path_st_f_new.shape[1]-1,num=(foot_path_st_f_new.shape[1]))
        st1 = np.arange(0, foot_path_st_f_new.shape[1])
        sw1 = np.arange(foot_path_st_f_new.shape[1], (foot_path_st_f_new.shape[1] + foot_path_sw_new_5th.shape[1]))
        st2 = np.arange((foot_path_st_f_new.shape[1] + foot_path_sw_new_5th.shape[1]), foot_path.shape[1])

        # expand our kinematics to achieve the proper stance/swing duty
        actual_num_swings_samps = len(sw1)
        actual_num_stance_samps = len(st1) + len(st2)

        # add samples to the stance phase to achieve the proper proportion of stance and swing time steps
        # assuming that you will never be moving fast enough to need to remove stance phase points
        additional_samps = math.floor((self.stance_duty * num_samps - actual_num_stance_samps) / (1 - self.stance_duty))
        desired_num_stance_samps = actual_num_stance_samps + additional_samps

        # indices of the stance phase now that resampled the kinematics
        st1_resample = np.linspace(st1[0], st1[-1], math.floor(desired_num_stance_samps / 2))
        st2_resample = np.linspace(st2[0], st2[-1], math.ceil(desired_num_stance_samps / 2))

        # interpolate the stance phase kinematics
        st1_kinematics = np.zeros((kinematics.shape[0], len(st1_resample)))
        foot_path_new_5th_st1 = np.zeros((3, len(st1_resample)))

        for v in range(kinematics.shape[0]):
            f = interpolate.interp1d(st1, kinematics[v, st1], kind="cubic")
            st1_kinematics[v, :] = f(st1_resample)
        for v in range(3):
            f = interpolate.interp1d(st1, foot_path[v][st1])
            foot_path_new_5th_st1[v, :] = f(st1_resample)

        st2_kinematics = np.zeros((kinematics.shape[0], len(st2_resample)))
        foot_path_new_5th_st2 = np.zeros((3, len(st2_resample)))
        for v in range(kinematics.shape[0]):
            f = interpolate.interp1d(st2, kinematics[v, st2])
            st2_kinematics[v, :] = f(st2_resample)
        for v in range(3):
            # Must resample the foot trajectory itself (MATLAB behavior),
            # not the joint-angle kinematics array.
            f = interpolate.interp1d(st2, foot_path[v, st2])
            foot_path_new_5th_st2[v, :] = f(st2_resample)

        st_kinematics = np.concatenate((st2_kinematics, st1_kinematics), axis=1)

        # swing phase kinematics
        sw_kinematics = kinematics[:,
                        range(foot_path_st_f_new.shape[1], foot_path_st_f_new.shape[1] + foot_path_sw_new_5th.shape[1])]
        # Use full swing polynomial so length matches sw_kinematics / IK timestep count.
        foot_path_new_5th_sw = foot_path_sw_new_5th

        # full step with stance and swing proportioned correctly
        kinematics_resample = np.concatenate((st1_kinematics, sw_kinematics, st2_kinematics), axis=1)
        foot_path_resample = np.concatenate((foot_path_new_5th_st1, foot_path_new_5th_sw, foot_path_new_5th_st2),
                                            axis=1)
        # Keep foot trajectory on the same time grid as joint resample (guards rare 102 vs 103 off-by-one).
        n_k = kinematics_resample.shape[1]
        n_f = foot_path_resample.shape[1]
        if n_f != n_k:
            t_old = np.linspace(0.0, 1.0, n_f)
            t_new = np.linspace(0.0, 1.0, n_k)
            foot_path_resample = np.vstack(
                [np.interp(t_new, t_old, foot_path_resample[v, :]) for v in range(3)]
            )
        # position of the body moving backward
        traj_dir = foot_path_resample[0][1] - foot_path_resample[0][0]
        if traj_dir < 0:
            aepid = np.argmax(foot_path_resample[0, :])
            pepid = np.argmin(foot_path_resample[0, :])
        else:
            aepid = np.argmin(foot_path_resample[0, :])
            pepid = np.argmax(foot_path_resample[0, :])
        # redefine the num_samps to match the new number of samples

        num_samps = kinematics_resample.shape[1]
        period = 1 / step_frequency
        # make a time vector to produce desired frequency
        t = np.arange(1, num_samps + 1)
        t = t / num_samps * period
        t_stance = t[range(0, desired_num_stance_samps)]

        # smooth out the disconnect between the start and end of the kinematics due to error creep with
        # a 3rd order polynomial interpolation
        num_pts = round(num_samps / 6)
        if num_pts % 2 != 0:
            num_pts = num_pts + 1
        p1 = int(num_pts / 2 - 1)
        p2 = int(num_pts / 2)
        q3 = lambda t, p: p[0] + p[1] * t + p[2] * t ** 2 + p[3] * t ** 3
        t0 = 0
        tf = num_pts
        t_smooth = np.linspace(t0, tf, num_pts)

        a3_13 = [[1, t0, t0 * t0, t0 ** 3], [0, 1, 2 * t0, 3 * t0 * t0], [1, tf, tf * tf, tf ** 3],
                 [0, 1, 2 * tf, 3 * tf * tf]]

        # fix to include the RF joints
        for k in range(3):
            end = kinematics_resample.shape[1]
            z0 = kinematics_resample[k, end - p1]
            vz0 = kinematics_resample[k, end - p1] - kinematics_resample[k, end - (p1 + 1)]
            zf = kinematics_resample[k, p2 + 1]
            vzf = kinematics_resample[k, p2 + 1] - kinematics_resample[k, p2]

            b3_13 = [z0, vz0, zf, vzf]
            f = np.linalg.solve(a3_13, b3_13)
            new_kin = q3(t_smooth, f)
            kinematics_resample[k, end - p1:end] = new_kin[:int(num_pts / 2) - 1]
            kinematics_resample[k, :p2] = new_kin[int(num_pts / 2):]

        return kinematics_resample, st_kinematics, sw_kinematics, aepid, pepid, foot_path_resample


