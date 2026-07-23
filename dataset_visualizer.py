import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Script just graphs each feature over time to visualize"""

#load dataset and extract individual features
raw_df = pd.read_csv("csv/gait_phase_LH_trajectories.csv")
time = raw_df["time"]

step = 1 #per 2ms
time_downsamp =time[::step]

LH_CTr_theta = raw_df["LH_CTr_theta"]
LH_TrF_theta = raw_df["LH_TrF_theta"]
LH_FTi_theta = raw_df["LH_FTi_theta"]

LH_CTr_vel = raw_df["LH_CTr_vel"]
LH_TrF_vel = raw_df["LH_TrF_vel"]
LH_FTi_vel = raw_df["LH_FTi_vel"]

GRF_x = raw_df["LH_GRF_x"]
GRF_y = raw_df["LH_GRF_y"]
GRF_z = raw_df["LH_GRF_z"]

figure, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10), sharex=True)
ax1, ax2, ax3, ax4, ax5, ax6, ax_x, ax_y, ax_z = axes.flatten()

ax1.plot(time_downsamp, LH_CTr_theta[::step], label="LH_CTr_theta")
ax1.set_ylabel('Radians (rads)')
ax1.set_title('LH_CTr_theta')

ax2.plot(time_downsamp, LH_TrF_theta[::step], label="LH_TrF_theta")
ax2.set_title('LH_TrF_theta')

ax3.plot(time_downsamp, LH_FTi_theta[::step], label="LH_FTi_theta")
ax3.set_title('LH_FTi_theta')

ax4.plot(time_downsamp, LH_CTr_vel[::step], label="LH_CTr_vel")
ax4.set_ylabel('Radians per sec (rads/s)')
ax4.set_title('LH_CTr_vel')

ax5.plot(time_downsamp, LH_TrF_vel[::step], label="LH_TrF_vel")
ax5.set_title('LH_TrF_vel')

ax6.plot(time_downsamp, LH_FTi_vel[::step], label="LH_FTi_vel")
ax6.set_title('LH_FTi_vel')

ax_x.plot(time_downsamp, GRF_x[::step], label="GRF_x")
ax_x.set_xlabel('time (s)')
ax_x.set_ylabel('Force (N)')
ax_x.set_title('GRF_x')

ax_y.plot(time_downsamp, GRF_y[::step], label="GRF_y")
ax_y.set_xlabel('time (s)')
ax_y.set_title('GRF_y')

ax_z.plot(time_downsamp, GRF_z[::step], label="GRF_z")
ax_z.set_xlabel('time (s)')
ax_z.set_title('GRF_z')
figure.suptitle('Neural Network Inputs')
plt.show()