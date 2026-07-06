import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Script just graphs each feature over time to visualize"""

#load dataset and extract individual features
raw_df = pd.read_csv("csv/gait_phase_LH_trajectories.csv")
time = raw_df["time"]

step = 1 #per second
time_downsamp =time[::step]

LH_CTr_pos = raw_df["LH_CTr_pos"]
LH_TrF_pos = raw_df["LH_TrF_pos"]
LH_FTi_pos = raw_df["LH_FTi_pos"]

LH_CTr_acc = raw_df["LH_CTr_acc"]
LH_TrF_acc = raw_df["LH_TrF_acc"]
LH_FTi_acc = raw_df["LH_FTi_acc"]

GRF_x = raw_df["LH_GRF_x"]
GRF_y = raw_df["LH_GRF_y"]
GRF_z = raw_df["LH_GRF_z"]

figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,10), sharex=True)
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

ax1.plot(time_downsamp, LH_CTr_pos[::step], label="LH_CTr_pos")
ax1.set_ylabel('Radians (rads)')
ax1.set_title('LH_CTr_pos')

ax2.plot(time_downsamp, LH_TrF_pos[::step], label="LH_TrF_pos")
ax2.set_title('LH_TrF_pos')

ax3.plot(time_downsamp, LH_FTi_pos[::step], label="LH_FTi_pos")
ax3.set_title('LH_FTi_pos')

ax4.plot(time_downsamp, LH_CTr_acc[::step], label="LH_CTr_acc")
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Radians per sec^2 (rads/s^2)')
ax4.set_title('LH_TrF_acc')

ax5.plot(time_downsamp, LH_TrF_acc[::step], label="LH_TrF_acc")
ax5.set_xlabel('Time (s)')
ax5.set_title('LH_TrF_acc')

ax6.plot(time_downsamp, LH_FTi_acc[::step], label="LH_FTi_acc")
ax6.set_xlabel('Time (s)')
ax6.set_title('LH_FTi_acc')

plt.show()

figure_GRF, axes_GRF = plt.subplots(nrows=1, ncols=3, figsize=(10,10), sharex=True)
ax_x, ax_y, ax_z = axes_GRF.flatten()
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
plt.show()