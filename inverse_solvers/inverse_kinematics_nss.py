import numpy as np
import mujoco
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import root
import copy
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from inverse_solvers.leg_height_con import leg_height_con
from scipy.optimize import minimize
from mujoco import viewer

class InverseKinematics:
    def __init__(self, num_rot, num_trans, leg_names, effectors, joint_names, p_base, stance_duty,
                 max_body_trans, max_body_rot, swing_duration, step_height, samps_per_step):

        self.leg_names = leg_names
        self.effectors = effectors
        self.joint_names = joint_names
        self.p_base = p_base
        self.stance_duty = stance_duty
        self.max_body_trans = max_body_trans
        self.max_body_rot = max_body_rot
        self.swing_duration = swing_duration
        self.stance_duration = self.swing_duration / (1 - self.stance_duty) * self.stance_duty
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
    def trans_rot_wrt_base(self, data, leg, num_trans, num_rot):
        leg_trans = np.linspace(-self.max_body_trans, self.max_body_trans, num_trans)
        leg_rot = np.linspace(-self.max_body_rot, self.max_body_rot, num_rot)
        trans_mat = np.meshgrid(leg_trans, leg_rot)[0]
        trans_mat = trans_mat.T

        rot_mat = np.zeros((num_trans, num_rot))
        for i in range(num_trans):
            if leg_trans[i] < 0:
                theta_lim = leg_trans[i] * self.max_body_rot / self.max_body_trans + self.max_body_rot
            else:
                theta_lim = -leg_trans[i] * self.max_body_rot / self.max_body_trans + self.max_body_rot
            rot_mat[i, :] = np.linspace(-theta_lim, theta_lim, num_rot)
        # rot_mat = rot_mat.T

        # find position of each foot wrt base point- will be used to find the foot velocity for each trans and rot velocity
        p_foot = data.site(self.effectors[leg]).xpos
        r_foot_wrt_base = p_foot - self.p_base

        return trans_mat, rot_mat, r_foot_wrt_base, leg_trans, leg_rot

    # stance_vec is the position of the foot of a certain leg at a specific rotation and translation
    def stance_vec_calc(self, r_foot_wrt_base, body_rot_per_step, body_trans_per_step):
        j_p = np.array(r_foot_wrt_base)
        psi = body_rot_per_step
        r = [body_trans_per_step, 0, 0]
        print("psi = " + str(psi) + ", r = " + str(r))
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

        # WHAT THE HELL all this is screwed up. #This only works if the stance_duty is 0.5. But it could be anything.

        swing_x_vel = (x_min - x_max) / self.stance_duration
        swing_x_acc = 0
        single_time_step = self.stance_duration / step_num
        swing_z_vel1 = (stance_vec[2, -1] - stance_vec[2, -2]) / single_time_step
        swing_z_vel2 = (stance_vec[2, 1] - stance_vec[2, 0]) / single_time_step

        b5_12z = np.array([[stance_vec[2, -1], swing_z_vel1, 0, self.step_height[leg], 0, 0]]).T
        b5_23z = np.array([[self.step_height[leg], 0, 0, stance_vec[2, 0], swing_z_vel2, 0]]).T
        b5_13x = np.array([[x_max, -swing_x_vel, swing_x_acc, x_min, -swing_x_vel, swing_x_acc]]).T

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

        diffs = np.diff(stance_vec[0, :])
        # If the stance_vec[0, :] is pretty much uniform, it is "invalid" because we can't use it for interpolation.
        # In this case, "uniform" means varies by less than a millimeter
        invalid = all(abs(diffs) < 1e-6)

        # interpolate z coordinates in swing based stance
        if invalid:
            y5_sw = np.zeros_like(x5_sw)
        elif np.linalg.norm(x5_sw) != 0:
            # Pchip only allows for strictly increasing monotonic series, need to check if strictly decreasing
            # This condition checks if stance_vec[0, :] increases monotonically.
            if all(np.diff(stance_vec[0, :]) > 0):
                interp = interpolate.PchipInterpolator(stance_vec[0, :], stance_vec[1, :], extrapolate=True)
                y5_sw = interp(x5_sw)
            elif all(np.diff(stance_vec[0, :]) < 0):
                interp = interpolate.PchipInterpolator(np.flip(stance_vec[0, :]), np.flip(stance_vec[1, :]), extrapolate=True)
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
            y5_sw = np.zeros_like(x5_sw)
        foot_path_sw_new_5th = np.zeros((3, self.num_samps_per_step))
        foot_path_sw_new_5th[0, :] = x5_sw
        foot_path_sw_new_5th[1, :] = y5_sw
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
                plt.plot(np.tile(foot_path[n, :], (3, )), label="axis " + str(n))
            plt.legend()
            plt.title("perfect leg")
            plt.ylabel("foot path (m)")
            plt.grid()
            plt.draw()
            plt.show()

        return foot_path, foot_path_st_f_new, foot_path_sw_new_5th

    # at a specific translation and rotation, use moore_penrose to solve for theta
    def moore_penrose(self, model, data, leg, r_foot_wrt_base, foot_path, start_and_end_same_point=False):

        # find active joints for the leg
        joint_id = []
        for joints in self.joint_names:
            if self.leg_names[leg] in joints:
                joint_id.append(model.joint(joints).id)
        j_qposaddr = model.jnt_qposadr[joint_id]
        qvel_addr = model.jnt_dofadr[joint_id]

        data.qpos[j_qposaddr] = 0  # This is the rest configuration
        mujoco.mj_forward(model, data)

        p_foot0 = copy.deepcopy(data.site(self.effectors[leg]).xpos)
        if any(np.isnan(p_foot0)):
            # This leg does not have a resting posture
            raise Exception("Leg with name \"" + self.leg_names[leg] + "\" has no rest posture defined, "
                            "so inverse kinematics cannot be calculated.")

        if np.linalg.norm(foot_path[:, 0]) > 1e-6:
            # Foot paths aren't centered at the foot's position at rest.
            1+1

        # r_foot_wrt_base[:] = p_foot - self.p_base

        step_frequency = 1 / (self.stance_duration + self.swing_duration)

        # figure out what IDs AEP and PEP are for this foot trajectory
        # use moore-penrose pseudo-inverse of the body-fram jacobian. Will find the joint velocities that move the foot
        # along the desired trajectories with the least joint motion possible
        # save position of the root body for comparison to foot positions

        # body_center = data.body("Thorax").xpos - self.p_base
        # end_effector = data.site(self.effectors[leg]).xpos
        # start_point = end_effector - self.p_base

        unique_solution = bool(len(j_qposaddr) == 3)
        num_steps = foot_path.shape[1]
        print("num_steps = " + str(num_steps))
        kinematics = np.empty((len(j_qposaddr), num_steps))
        r_foot = np.zeros((3, num_steps))
        # r_foot[:, 0] is deliberately 0.
        r_foot_imp = np.zeros((3, num_steps))
        kinematics[:, 0] = 0
        kinematics_imp = np.empty((len(j_qposaddr), num_steps))
        kinematics_imp[:, 0] = 0
        # orig_joints = joint_id

        # compute joint kinematics for the desired motion
        i = 1
        ftol = 1e-6
        to_continue_stepping = True
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        site_id = model.site(self.effectors[leg]).id

        j_max = int(1e4)
        if unique_solution:
            print("Searching for unique solution.")
            while i < num_steps and to_continue_stepping:
                """
                dx = foot_path[:, i] - foot_path[:, i - 1]

                candidate_config = kinematics_imp[:, i - 1]
                converged = False
                to_continue_searching = True
                j = 0
                f_list = np.empty((3, j_max))
                x_list = np.empty((len(joint_id), j_max))
                jac_list = np.empty((3, len(joint_id), j_max))

                x_list[:, 0] = candidate_config
                jac, p = self.set_config_get_jacobian(model, data, joint_id, site_id, p_foot0, candidate_config)
                # f must include the actual foot position in the candidate_config to ensure convergence.
                f_candidate = foot_path[:, i] - p
                f_list[:, 0] = f_candidate
                jac_list[:, :, 0] = jac
                delta_x = np.zeros((len(joint_id),))

                while to_continue_searching:
                    to_direction_search = False
                    if np.linalg.norm(f_list[:, j]) < 1e-9:
                        converged = True
                        to_continue_searching = False
                        x_final = x_list[:, j]
                        f_final = f_list[:, j]
                        print("solved, f: " + str(f_final) + ", config: " + str(x_final) + ", loops: " + str(j))
                        print(np.linalg.cond(jac_list[:, :, j] @ jac_list[:, :, j].T))
                    elif j >= j_max:
                        converged = False
                        to_continue_searching = False
                        x_final = np.full_like(x_list[:, j], np.nan)
                        f_final = np.full_like(f_list[:, j], np.nan)
                        print("maxit, f: " + str(f_final) + ", config: " + str(x_final) + ", loops: " + str(j))
                        print(np.linalg.cond(jac_list[:, :, j].T @ jac_list[:, :, j]))
                    else:
                        converged = False
                        to_continue_searching = True
                        to_direction_search = True

                    if to_continue_searching and to_direction_search:
                        # jac, p = self.set_config_get_jacobian(model, data, joint_id, site_id, p_foot0, candidate_config)
                        # f must include the actual foot position in the candidate_config to ensure convergence.
                        # f = foot_path[:, i] - p
                        # Take a Newton step, with the understanding that the manipulator jacobian is df/dtheta.
                        delta_x = np.linalg.pinv(jac_list[:, :, j]) @ f_list[:, j]

                        # Evaluate the function at the candidate solution
                        jac, p = self.set_config_get_jacobian(model, data, joint_id, site_id, p_foot0,
                                                              x_list[:, j] + delta_x)
                        f_candidate = foot_path[:, i] - p
                        # If the candidate configuration x_list + delta_x moved f further from 0, we need to "line search"
                        while bool(np.linalg.norm(f_candidate) > np.linalg.norm(f_list[:, j])):
                            delta_x = delta_x / 2
                            print("j = " + str(j) + ", delta_x = " + str(delta_x))

                            # Evaluate the function at the candidate solution
                            jac, p = self.set_config_get_jacobian(model, data, joint_id, site_id, p_foot0,
                                                                  x_list[:, j] + delta_x)
                            f_candidate = foot_path[:, i] - p

                            if np.linalg.norm(delta_x) < 1e-9:
                                to_line_search = False
                                to_continue_searching = False
                                x_final = x_list[:, j]
                                f_final = f_list[:, j]
                                print("stalled, f: " + str(f_final) + ", config: " + str(x_final) + ", loops: " + str(j))
                                print(np.linalg.cond(jac_list[:, :, j].T @ jac_list[:, :, j]))

                    #Are we sure these are the latest, best estimates?
                    try:
                        f_list[:, j + 1] = f_candidate
                    except NameError:
                        print("f_candidate not assigned.")
                        1+1

                    try:
                        x_list[:, j + 1] = x_list[:, j] + delta_x
                    except NameError:
                        print("delta_x not assigned.")
                        1+1

                    jac_list[:, :, j + 1] = jac
                    # If foot position error is close to 0, quit searching and output some info.

                    j += 1

                kinematics_imp[:, i] = x_final
                r_foot_imp[:, i] = p
                """

                """ BELOW, REPLACE WITH ROOT """
                # Use root to solve inverse kinematics, compare to custom script.

                f_position = lambda x: foot_path[:, i] - self.set_config_get_jacobian(model, data, j_qposaddr, qvel_addr, site_id, p_foot0, x)[1]
                g_jacobian = lambda x: -self.set_config_get_jacobian(model, data, j_qposaddr, qvel_addr, site_id, p_foot0, x)[0]

                options = {"maxfev": int(1e3), "xtol": 1e-10}
                result = root(f_position, kinematics_imp[:, i - 1], method='hybr', jac=g_jacobian, options=options)
                kinematics_imp[:, i] = result.x
                if result.status != 1:
                    if np.linalg.norm(result.fun) > ftol:
                        print("Result " + str(result.status) + ": " + result.message + ", " + str(result.fun))
                        start_and_end_same_point = True
                    """
                    with viewer.launch_passive(model, data) as v:
                        while v.is_running():
                            # mujoco.mj_step(model, data)
                            v.sync()
                    """
                    1+1
                r_foot_imp[:, i] = foot_path[:, i] - result.fun

                i += 1

        else:
            # Not a unique solution, so we need a minimization approach
            while i < num_steps:
                g_distance = lambda x: np.dot(x, x)
                bounds = [(-np.pi, np.pi)] * len(j_qposaddr)
                options = {"ftol": 1e-10, "maxiter": 1e3}
                # constrain the leg posture for the z (height)

                cons = ({'type': "eq", "fun": lambda x: leg_height_con(x, site_id, j_qposaddr, foot_path[:, i], data, model) - p_foot0})
                result = minimize(g_distance, kinematics_imp[:, i-1], method='SLSQP', bounds=bounds, constraints=cons, options=options)
                kinematics_imp[:, i] = result.x
                if any(abs(result.x) > np.pi):
                    print(result.x)
                    print("Max constraint violation of " + str(cons["fun"](result.x)) + " after " + str(
                        result.nit) + " iterations.")
                    1+1
                # print("Max constraint violation of " + str(cons["fun"](result.x)) + " after " + str(result.nit) + " iterations.")
                jac_end, p_final = self.set_config_get_jacobian(model, data, j_qposaddr, qvel_addr, site_id, p_foot0, kinematics_imp[:, i])
                r_foot_imp[:, i] = p_final

                i += 1


        if start_and_end_same_point:
            # print("error: ")
            # print(np.linalg.norm(kinematics[:, -1] - kinematics[:, 0]))
            # If the norm of the error is too big, we need to reduce dt or something.
            # Make adaptive time-step moore-penrose?

            #I could, but easier would be to:
            # define callable Jacobian function to act as "f(x) = dx/dt"
            # use R-K45 to integrate. scipy.integrate.RK45
            err = (r_foot[:, -1] - r_foot[:, 0])
            print("vec error for p_inv of Jacobian = " + str(err))
            err_imp = (r_foot_imp[:, -1] - r_foot_imp[:, 0])
            print("vec error for N-R with Jacobian = " + str(err_imp))

            """plt.figure()
            for n in range(np.shape(foot_path)[0]):
                # plt.plot(np.tile(foot_path[n, :], (3,)), label="axis " + str(n))
                plt.plot(foot_path[n, :], label="com, axis " + str(n))
            for n in range(np.shape(r_foot_imp)[0]):
                # plt.plot(np.tile(foot_path[n, :], (3,)), label="axis " + str(n))
                plt.plot(r_foot_imp[n, :], linestyle="dashed", label="act, axis " + str(n))
            plt.legend()
            plt.title("inside moore-penrose")
            plt.ylabel("foot path (m)")
            plt.grid()
            plt.draw()

            plt.figure()
            for n in range(np.shape(kinematics_imp)[0]):
                # plt.plot(np.tile(foot_path[n, :], (3,)), label="axis " + str(n))
                plt.plot(np.tile(kinematics_imp[n, :], 3), label="imp joint " + str(n))
            plt.legend()
            plt.title("inside moore-penrose")
            plt.ylabel("joint angles (rad)")
            plt.grid()
            plt.draw()

            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(foot_path[0, :], foot_path[1, :], foot_path[2, :], label="prescribed path")
            ax.plot(r_foot_imp[0, :], r_foot_imp[1, :], r_foot_imp[2, :], linestyle="dashed", label="actual path")
            ax.legend()
            ax.set(xlabel="x", ylabel="y")
            ax.axis("equal")
            ax.grid()
            plt.show()
            """

        return kinematics_imp, step_frequency

    def set_config_get_jacobian(self, model, data, joint_id, joint_vel_id, site_id, p_foot0, config):
        if not all(np.equal(config, data.qpos[joint_id])):
            # If this config is new, assign it to qpos and update the model's kinematics.
            data.qpos[joint_id] = config
            mujoco.mj_kinematics(model, data)

        jacp = np.zeros((3, model.nv))
        mujoco.mj_kinematics(model, data)
        mujoco.mj_jacSite(model, data, jacp, jacr=None, site=site_id)
        return copy.deepcopy(jacp[:, joint_vel_id]), copy.deepcopy(data.site(site_id).xpos) - p_foot0

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

        # compute leg configuration in pep and aep
        pep = st1_kinematics[:, -1]
        aep = st2_kinematics[:, 0]

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

        return kinematics_resample, st_kinematics, sw_kinematics, aep, pep, foot_path_resample


