import numpy as np
import mujoco

class InverseDynamics():
    def __init__(self,model,data,full_kinematics,body_names,leg_names,joint_names,swing_duration,
                     stance_duty,contra_phase,ipsal_phase):
        self.model=model
        self.data=data
        self.full_kinematics=full_kinematics
        self.body_names=body_names
        self.leg_names=leg_names
        self.joint_names=joint_names
        self.swing_duration=swing_duration
        self.stance_duty=stance_duty
        self.contra_phase=contra_phase
        self.ipsal_phase=ipsal_phase
        
        self.samps_per_step = full_kinematics[0, 0, 0].shape[1]
        self.time_vec = np.array(range(0, self.samps_per_step))
        self.step_period = (swing_duration / (1 - stance_duty))
        self.dt=self.step_period/self.samps_per_step

    def _joint_ids_for_leg(self, leg_idx):
        ids = []
        leg_tag = self.leg_names[leg_idx]
        for joint_name in self.joint_names:
            if leg_tag in joint_name:
                ids.append(self.model.joint(joint_name).id)
        return ids

    # calculate the omegas for each joint and velocities of each body by averaging over the previous and future timesteps
    def vel_acc(self, leg, trans_vec, rot_vec):
        # Match vel_and_acc expectation: explicit central differences with periodic wrap.
        qpos_leg = np.asarray(self.full_kinematics[leg, trans_vec, rot_vec], dtype=np.float64)
        num_joints, num_samples = qpos_leg.shape
        qvel_leg = np.zeros((num_joints, num_samples), dtype=np.float64)
        qacc_leg = np.zeros((num_joints, num_samples), dtype=np.float64)

        for k in range(num_samples):
            km1 = (k - 1) % num_samples
            kp1 = (k + 1) % num_samples
            qvel_leg[:, k] = (qpos_leg[:, kp1] - qpos_leg[:, km1]) / (2.0 * self.dt)
            qacc_leg[:, k] = (qvel_leg[:, kp1] - qvel_leg[:, km1]) / (2.0 * self.dt)

        return qvel_leg, qacc_leg

    def solve_inverse_dynamics_with_contact_plane(self,trans_idx,rot_idx,foot_path,vel,acc,plane_geom_name="ground"):
        num_legs = len(self.leg_names)
        tau = np.zeros((num_legs, self.samps_per_step), dtype=object)
        grf_raw = np.zeros((num_legs, self.samps_per_step, 3))
        leg_has_plane_contact = np.zeros((num_legs, self.samps_per_step), dtype=bool)

        leg_joint_ids = [self._joint_ids_for_leg(leg_idx) for leg_idx in range(num_legs)]
        leg_qpos = [self.full_kinematics[leg_idx, trans_idx, rot_idx] for leg_idx in range(num_legs)]
        if len(vel) != num_legs or len(acc) != num_legs:
            raise ValueError("vel and acc must each contain one entry per leg.")


        plane_geom_id = self.model.geom(plane_geom_name).id
        tip_body_ids = [self.model.body(f"{name}_Tip").id for name in self.leg_names]

        for t in range(self.samps_per_step):
            # Clear dynamic states each frame so only this step's kinematics drive dynamics.
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0

            for leg_idx in range(num_legs):
                jids = leg_joint_ids[leg_idx]
                qpos_leg = leg_qpos[leg_idx]
                self.data.qpos[jids] = qpos_leg[:, t]

            # Build position-dependent/contact state first.
            mujoco.mj_forward(self.model, self.data)

            for c_idx in range(self.data.ncon):
                contact = self.data.contact[c_idx]
                geom_a = contact.geom1
                geom_b = contact.geom2
                if geom_a != plane_geom_id and geom_b != plane_geom_id:
                    continue

                other_geom = geom_b if geom_a == plane_geom_id else geom_a
                other_body_id = self.model.geom_bodyid[other_geom]
                if other_body_id not in tip_body_ids:
                    continue

                wrench = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, c_idx, wrench)
                # Use full 3D contact force so Fx/Fy/Fz are preserved.
                frame = contact.frame.reshape(3, 3)
                force_world = frame @ wrench[:3]

                leg_idx = tip_body_ids.index(other_body_id)
                # Contact force is defined on geom1. Flip sign when plane is geom1.
                if geom_a == plane_geom_id:
                    force_world = -force_world
                grf_raw[leg_idx, t, :] += force_world
                leg_has_plane_contact[leg_idx, t] = True

            # Apply chosen motion terms after mj_forward so they are not overwritten.
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            for leg_idx in range(num_legs):
                jids = leg_joint_ids[leg_idx]
                qvel_leg = np.asarray(vel[leg_idx], dtype=np.float64)
                qacc_leg = np.asarray(acc[leg_idx], dtype=np.float64)
                expected_shape = (len(jids), self.samps_per_step)
                if qvel_leg.shape != expected_shape or qacc_leg.shape != expected_shape:
                    raise ValueError(f"vel/acc for leg {leg_idx} must have shape {expected_shape}.")
                self.data.qvel[jids] = qvel_leg[:, t]
                self.data.qacc[jids] = qacc_leg[:, t]

            mujoco.mj_inverse(self.model, self.data)

            # Motor torque required for this timestep from inverse dynamics.
            for leg_idx in range(num_legs):
                jids = leg_joint_ids[leg_idx]
                tau[leg_idx, t] = self.data.qfrc_inverse[jids].copy()

        # tip_z: row 2 of each leg's (3, T) foot_path_resample; shape (num_legs, T).
        """
        tip_z = np.vstack([np.asarray(foot_path[i], dtype=np.float64)[2, :] for i in range(num_legs)])
        z_min = np.min(tip_z, axis=1, keepdims=True)
        contact_bandwidth_m = float(0.1) * 1e-3
        plane_z = float(contact_plane_z)
        denom = plane_z - z_min
        denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
        contact_scale = np.clip((tip_z - z_min) / denom, 0.0, 1.0)
        """

        # Scale GRF so it is zero at z_min (ground) and max at contact_plane_z.
        #grf = grf_raw * contact_scale[:, :, None]

        grf=grf_raw

        return tau, grf

    def estimate_torque_from_grf(self, trans_idx, rot_idx, grf):
        # Quasi-static estimate using virtual work: tau = J^T * F at each tip.
        num_legs = len(self.leg_names)
        tau_est = np.empty((num_legs, self.samps_per_step), dtype=object)
        leg_joint_ids = [self._joint_ids_for_leg(leg_idx) for leg_idx in range(num_legs)]
        leg_qpos = [self.full_kinematics[leg_idx, trans_idx, rot_idx] for leg_idx in range(num_legs)]
        tip_body_ids = [self.model.body(f"{name}_Tip").id for name in self.leg_names]

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)

        for t in range(self.samps_per_step):
            # Rebuild full-body posture for this gait sample.
            for leg_idx in range(num_legs):
                jids = leg_joint_ids[leg_idx]
                self.data.qpos[jids] = leg_qpos[leg_idx][:, t]

            mujoco.mj_forward(self.model, self.data)

            for leg_idx in range(num_legs):
                force_world = np.asarray(grf[leg_idx, t], dtype=np.float64)
                jacp[:] = 0.0
                mujoco.mj_jacBodyCom(self.model, self.data, jacp, None, tip_body_ids[leg_idx])
                j_leg = jacp[:, leg_joint_ids[leg_idx]]
                tau_est[leg_idx, t] = j_leg.T @ force_world

        return tau_est
