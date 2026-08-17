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

        GRF_z = raw_df["LH_GRF_z"]

        return LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, GRF_z
     
    def plot_features(time, LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel):
        figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(20,20), sharex=True)
        ax1, ax2, ax3, ax4, ax5, ax6, = axes.flatten()

        ax1.plot(time, LH_CTr_theta, label="LH_CTr_theta")
        ax1.set_ylabel('Radians (rads)')
        ax1.set_title('LH_CTr_theta')

        ax2.plot(time, LH_TrF_theta, label="LH_TrF_theta")
        ax1.set_ylabel('Radians (rads)')
        ax2.set_title('LH_TrF_theta')

        ax3.plot(time, LH_FTi_theta, label="LH_FTi_theta")
        ax1.set_ylabel('Radians (rads)')
        ax3.set_title('LH_FTi_theta')

        ax4.plot(time, LH_CTr_vel, label="LH_CTr_vel")
        ax4.set_ylabel('Radians per sec (rads/s)')
        ax4.set_title('LH_CTr_vel')

        ax5.plot(time, LH_TrF_vel, label="LH_TrF_vel")
        ax4.set_ylabel('Radians per sec (rads/s)')
        ax5.set_title('LH_TrF_vel')

        ax6.plot(time, LH_FTi_vel, label="LH_FTi_vel")
        ax4.set_ylabel('Radians per sec (rads/s)')
        ax6.set_title('LH_FTi_vel')

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
        raw_data_GRF = raw_data_GRF.to_numpy() # convert to numpy for plotting
        plt.figure(figsize=(10, 5))

        plt.plot(aligned_time, raw_data_GRF)
        plt.scatter(aligned_time[swing_indices], raw_data_GRF[swing_indices], color='red', label='Predicted Swing States')
        
        plt.title("GRF z-axis")
        plt.xlabel('time (s)')
        plt.legend()
        plt.suptitle('LH leg Gait Classification of a Single Cycle')
        plt.show()







