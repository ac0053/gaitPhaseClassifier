import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_features(time, LF_TrF_theta, LF_FTi_theta, LF_TrF_vel, LF_FTi_vel, LF_TrF_acc,LF_FTi_acc, LF_TrF_cmd, LF_FTi_cmd, LF_GRF_x, LF_GRF_y, LF_GRF_z):
        figure, axes = plt.subplots(nrows=11, ncols=1, figsize=(15,10), sharex=True)
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11 = axes.flatten()

        ax1.plot(time, LF_TrF_theta, label="LF_TrF_theta")
        ax1.set_ylabel('Radians (rads)', fontsize=5)
        ax1.set_title('LF_TrF_theta', fontsize=10)

        ax2.plot(time, LF_FTi_theta, label="LF_FTi_theta")
        ax2.set_ylabel('Radians (rads)', fontsize=5)
        ax2.set_title('LF_FTi_theta', fontsize=10)

        ax3.plot(time, LF_TrF_vel, label="LF_TrF_vel")
        ax3.set_ylabel('Radians/s (rads/s)', fontsize=5)
        ax3.set_title('LF_TrF_vel', fontsize=10)

        ax4.plot(time, LF_FTi_vel, label="LF_FTi_vel")
        ax4.set_ylabel('Radians/s (rads/s)', fontsize=5)
        ax4.set_title('LF_FTi_vel', fontsize=10)

        ax5.plot(time, LF_TrF_acc, label="LF_TrF_acc")
        ax5.set_ylabel('Radians/s^2 (rads/s^2)', fontsize=5)
        ax5.set_title('LF_TrF_acc', fontsize=10)

        ax6.plot(time, LF_FTi_acc, label="LF_FTi_acc")
        ax6.set_ylabel('Radians/s^2 (rads/s^2)', fontsize=5)
        ax6.set_title('LF_FTi_acc', fontsize=10)

        ax7.plot(time, LF_TrF_cmd, label="LF_TrF_cmd")
        ax7.set_ylabel('Radians (rads)', fontsize=5)
        ax7.set_title('LF_TrF_cmd', fontsize=10)

        ax8.plot(time, LF_FTi_cmd, label="LF_FTi_cmd")
        ax8.set_ylabel('Radians (rads)', fontsize=5)
        ax8.set_title('LF_FTi_cmd', fontsize=10)

        ax9.plot(time, LF_GRF_x, label="LF_GRF_x")
        ax9.set_ylabel('Newton (N)', fontsize=5)
        ax9.set_title('LF_GRF_x', fontsize=10)

        ax10.plot(time, LF_GRF_y, label="LF_GRF_y")
        ax10.set_ylabel('Newton (N)', fontsize=5)
        ax10.set_title('LF_GRF_y', fontsize=10)

        ax11.plot(time, LF_GRF_z, label="LF_GRF_z")
        ax11.set_ylabel('Newton (N)', fontsize=5)
        ax11.set_xlabel('Time (s)')
        ax11.set_title('LF_GRF_z', fontsize=10)

        

        figure.suptitle('Neural Network Features (Inputs)')
        plt.show()

raw_df = pd.read_csv("csv_trajectory_datasets/master_trajectories_tran0_rot0.csv") # forward gait
TrFtheta = raw_df["LF_TrF_angle"]
FTitheta = raw_df["LF_FTi_angle"]
TrFvel = raw_df["LF_TrF_vel"]
FTivel = raw_df["LF_FTi_vel"]
TrFacc = raw_df["LF_TrF_acc"]
FTiacc = raw_df["LF_FTi_acc"]
TrFcmd = raw_df["LF_TrF_cmd"]
FTicmd = raw_df["LF_FTi_cmd"]
time = raw_df["timestamp"]
TorqueTrochCSx = raw_df["LF_Tip_GRF_x"]
TorqueTrochCSy = raw_df["LF_Tip_GRF_y"]
TorqueTrochCSz = raw_df["LF_Tip_GRF_z"]


plot_features(time, TrFtheta, FTitheta, TrFvel, FTivel, TrFacc, FTiacc, TrFcmd, FTicmd, TorqueTrochCSx, TorqueTrochCSy, TorqueTrochCSz)