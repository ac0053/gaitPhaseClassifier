import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_features(time, LF_TrF_theta, LF_FTi_theta, LF_TrF_cmd, LF_FTi_cmd, LF_Torque_Troch_CS_x, LF_Torque_Troch_CS_y, LF_Torque_Troch_CS_z, LF_Torque_Tibia_CS_x, LF_Torque_Tibia_CS_y, LF_Torque_Tibia_CS_z):
        figure, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20), sharex=True)
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = axes.flatten()

        ax1.plot(time, LF_TrF_theta, label="LF_TrF_theta")
        ax1.set_ylabel('Radians (rads)')
        ax1.set_title('LF_TrF_theta')

        ax2.plot(time, LF_FTi_theta, label="LF_FTi_theta")
        ax2.set_ylabel('Radians (rads)')
        ax2.set_title('LF_FTi_theta')

        ax3.plot(time, LF_TrF_cmd, label="LF_TrF_cmd")
        ax3.set_ylabel('Radians (rads)')
        ax3.set_title('LF_TrF_cmd')

        ax4.plot(time, LF_FTi_cmd, label="LF_FTi_cmd")
        ax4.set_ylabel('Radians (rads)')
        ax4.set_title('LF_FTi_cmd')

        ax5.plot(time, LF_Torque_Troch_CS_x, label="LF_Torque_Troch_CS_x")
        ax5.set_ylabel('Newton (N)')
        ax5.set_title('LF_InternalForce_Troch_CS_x')

        ax6.plot(time, LF_Torque_Troch_CS_y, label="LF_Torque_Troch_CS_y")
        ax6.set_ylabel('Newton (N): Internal Force')
        ax6.set_title('LF_InternalForce_Troch_CS_y')

        ax7.plot(time, LF_Torque_Troch_CS_z, label="LF_Torque_Troch_CS_z")
        ax7.set_ylabel('Newton (N): Internal Force')
        ax7.set_title('LF_InternalForce_Troch_CS_z')

        ax8.plot(time, LF_Torque_Tibia_CS_x, label="LF_Torque_Tibia_CS_x")
        ax8.set_ylabel('Newton (N): Internal Force')
        ax8.set_title('LF_InternalForce_Tibia_CS_x')

        ax9.plot(time, LF_Torque_Tibia_CS_y, label="LF_Torque_Tibia_CS_y")
        ax9.set_ylabel('Newton (N): Internal Force')
        ax9.set_title('LF_InternalForce_Tibia_CS_y')

        ax10.plot(time, LF_Torque_Tibia_CS_z, label="LF_Torque_Tibia_CS_z")
        ax10.set_ylabel('Newton (N): Internal Force')
        ax10.set_title('LF_InternalForce_Tibia_CS_z')

        figure.suptitle('Neural Network Features (Inputs)')
        plt.show()

raw_df = pd.read_csv("csv_trajectory_datasets/master_trajectories_tran0_rot0.csv") # forward gait
TrFtheta = raw_df["LF_TrF_theta"]
FTitheta = raw_df["LF_FTi_theta"]
TrFcmd = raw_df["LF_TrF_cmd"]
FTicmd = raw_df["LF_FTi_cmd"]
time = raw_df["time"]
TorqueTrochCSx = raw_df["LF_InternalForce_Troch_CS_x"]
TorqueTrochCSy = raw_df["LF_InternalForce_Troch_CS_y"]
TorqueTrochCSz = raw_df["LF_InternalForce_Troch_CS_z"]

TorqueTibiaCSx = raw_df["LF_InternalForce_Tibia_CS_x"]
TorqueTibiaCSy = raw_df["LF_InternalForce_Tibia_CS_y"]
TorqueTibiaCSz = raw_df["LF_InternalForce_Tibia_CS_z"]

plot_features(time, TrFtheta, FTitheta, TrFcmd, FTicmd, TorqueTrochCSx, TorqueTrochCSy, TorqueTrochCSz, TorqueTibiaCSx, TorqueTibiaCSy, TorqueTibiaCSz)