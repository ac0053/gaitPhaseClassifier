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

        GRF_x = raw_df["LH_GRF_x"]
        GRF_y = raw_df["LH_GRF_y"]
        GRF_z = raw_df["LH_GRF_z"]
        return LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, GRF_x, GRF_y, GRF_z
     
    def plot_features(time, LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, GRF_x, GRF_y, GRF_z):
        figure, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10), sharex=True)
        ax1, ax2, ax3, ax4, ax5, ax6, ax_x, ax_y, ax_z = axes.flatten()

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

        ax_x.plot(time, GRF_x, label="GRF_x")
        ax_x.set_xlabel('time (s)')
        ax_x.set_ylabel('Force (N)')
        ax_x.set_title('GRF_x')

        ax_y.plot(time, GRF_y, label="GRF_y")
        ax_y.set_xlabel('time (s)')
        ax_y.set_title('GRF_y')

        ax_z.plot(time, GRF_z, label="GRF_z")
        ax_z.set_xlabel('time (s)')
        ax_z.set_title('GRF_z')

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

    def plot_labeledTheta(time, raw_data_theta, SEQ_LENGTH, labels):
        aligned_time = time[SEQ_LENGTH:] # takes off first SEQ_LENGTH number of samples from start to match amout of time steps after sequencing features
        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
        ax1, ax2, ax3= axes.flatten()

        scatter_1 = ax1.scatter(aligned_time, raw_data_theta[SEQ_LENGTH:,0], c=labels)
        ax1.set_title("CTi angles")
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('joint angle (rads)')

        scatter_2 = ax2.scatter(aligned_time, raw_data_theta[SEQ_LENGTH:, 1], c=labels)
        ax2.set_title("TrF angles")
        ax2.set_xlabel('joint angle (s)')

        scatter_3 = ax3.scatter(aligned_time, raw_data_theta[SEQ_LENGTH:, 2], c=labels)
        ax3.set_title("FTi angles")
        ax3.set_xlabel('time (s)')

        cbar = figure.colorbar(scatter_3, ax=ax3, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Swing', 'Stance'])

        figure.suptitle('LH leg Gait Classification of a Single Cycle')
        plt.show()

    def plot_labeledGRF(time, raw_data_GRF, SEQ_LENGTH, labels):
        aligned_time = time[SEQ_LENGTH:] # takes off first SEQ_LENGTH number of samples from start to match amout of time steps after sequencing features
        figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
        ax4, ax5, ax6= axes.flatten()

        scatter_4 = ax4.scatter(aligned_time, raw_data_GRF[SEQ_LENGTH:, 0], c=labels, cmap='plasma')
        ax4.set_title("GRF x-axis")
        ax4.set_xlabel('time (s)')
        ax4.set_ylabel('force (N)')

        scatter_5 = ax5.scatter(aligned_time, raw_data_GRF[SEQ_LENGTH:, 1], c=labels, cmap='plasma')
        ax5.set_title("GRF y-axis")
        ax5.set_xlabel('time (s)')

        scatter_6 = ax6.scatter(aligned_time, raw_data_GRF[SEQ_LENGTH:, 2], c=labels, cmap='plasma')
        ax6.set_title("GRF z-axis")
        ax6.set_xlabel('time (s)')
        cbar = figure.colorbar(scatter_6, ax=ax6, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Swing', 'Stance'])
        figure.suptitle('LH leg Gait Classification of a Single Cycle')
        plt.show()







