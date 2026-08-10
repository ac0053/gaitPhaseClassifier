import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    pass

    def extract_features(raw_df):
        # load dataset and extract individual features
        LH_CTr_theta = raw_df["LH_CTr_theta"]
        LH_TrF_theta = raw_df["LH_TrF_theta"]
        LH_FTi_theta = raw_df["LH_FTi_theta"]

        LH_CTr_vel = raw_df["LH_CTr_vel"]
        LH_TrF_vel = raw_df["LH_TrF_vel"]
        LH_FTi_vel = raw_df["LH_FTi_vel"]

        effector_acc = raw_df["LH_Tip_linear_acc_z"]
        GRF_x = raw_df["LH_GRF_x"]

        return LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, effector_acc, GRF_x
     
    def plot_features(time, LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, effector_acc):
        figure, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10), sharex=True)
        ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axes.flatten()

        ax1.plot(time, LH_CTr_theta, label="LH_CTr_theta")
        ax1.set_ylabel('Radians (rads)')
        ax1.set_title('LH_CTr_theta')

        ax2.plot(time, LH_TrF_theta, label="LH_TrF_theta")
        ax2.set_title('LH_TrF_theta')

        ax3.plot(time, LH_FTi_theta, label="LH_FTi_theta")
        ax3.set_title('LH_FTi_theta')

        ax4.plot(time, LH_CTr_vel, label="LH_CTr_vel")
        ax4.set_ylabel('Radians per sec (rads/s)')
        ax4.set_title('LH_CTr_vel')

        ax5.plot(time, LH_TrF_vel, label="LH_TrF_vel")
        ax5.set_title('LH_TrF_vel')

        ax6.plot(time, LH_FTi_vel, label="LH_FTi_vel")
        ax6.set_title('LH_FTi_vel')

        ax7.plot(time, effector_acc, label="LH_Tip_linear_acc_z")
        ax7.set_ylabel('Acceleration (m/s^2)')
        ax7.set_title('LH_Tip_linear_acc_z')

        figure.suptitle('Neural Network Features (Inputs)')
        plt.show()

    def plot_rangeFrac_example(theta_frac):
        figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(10,15))
        ax7, ax8, ax9, ax10 = axes.flatten()

        ax7.plot(theta_frac[:, :, 0])
        ax7.set_title("CTi Joint Fractionation")
        ax7.set_xlabel('time steps (per 2 ms)')
        ax7.set_ylabel('magnitude')

        ax8.plot(theta_frac[:, :, 1])
        ax8.set_title("TrF Joint Fractionation")
        ax8.set_xlabel('time steps (per 2 ms)')

        ax9.plot(theta_frac[:, :, 2])
        ax9.set_title("FTi Joint Fractionation")
        ax9.set_xlabel('time steps (per 2 ms)')

        ax10.plot(theta_frac[:, -1, :])
        ax10.set_title("All 3 Joints Fractionation")
        ax10.set_xlabel('time steps (per 2 ms)')
        figure.suptitle("Range Fractionated Joint Angles")
        plt.show()

    def plot_labeledGRF(raw_df, raw_data_GRF, SEQ_LENGTH, swing_indices):
        aligned_time = raw_df["time"].values[SEQ_LENGTH:] # takes off first SEQ_LENGTH number of samples from start to match amout of time steps after sequencing features
        plt.figure(figsize=(10, 5))
        GRF_z = raw_data_GRF[SEQ_LENGTH:]

        plt.plot(aligned_time, GRF_z)
        plt.scatter(aligned_time[swing_indices-1], GRF_z[swing_indices-1], color='red', label='Predicted Swing States')
        plt.set_title("GRF z-axis")
        plt.set_xlabel('time (s)')
        plt.legend()
        plt.suptitle('LH leg Gait Classification of a Single Cycle')
        plt.show()







