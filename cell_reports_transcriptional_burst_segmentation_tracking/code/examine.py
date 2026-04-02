# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:42:19 2024

@author: cdsyou
"""

#%%

import napari
import skimage.io as io
import numpy as np
import pandas as pd
# import pickle
import matplotlib.pyplot as plt
# import seaborn as sns
import os
import seaborn as sns 

#%%

import sys

sys.path.append(r"G:\Other computers\My Laptop\lab\paper\general_utils")

from set_plotting_parameters import set_style
set_style()

#%%

# fig_dir = r"C:\Users\cdsyou\Desktop\TEMPORARY_TEST_20260109\JOINING_EVERYTHING_TOGETHER\figs"


#%%

# mv_parent_path = r"C:\Users\cdsyou\Desktop\TEMPORARY_TEST_20260109\JOINING_EVERYTHING_TOGETHER\mvs_out"
# field_of_interest = "col5_row3_field01"
filename = 'frame0_y309_x1069_track60_len216_joined_track'

#%%
# Uncomment if looking at manually segmented bursts
# man_bench = os.path.join(mv_parent_path,'manual_benchmarking')
# manual_track = mv_parent_path.split('\\')[4] + '__' + field_of_interest + '__' + filename + '.csv'
# manual_track_path = os.path.join(man_bench,manual_track)

# manual_trk = pd.read_csv(manual_track_path,index_col=0)


#%%
# field_of_interest_out =  field_of_interest + '_out'

nuc_id = filename.replace('_joined_track','')

start = int(filename.split('_')[0].replace('frame',''))

# out_dir = os.path.join(mv_parent_path, rf"out\{field_of_interest_out}")
out_dir = r"C:\Users\cdsyou\Documents\rab7a_analysed\CV7000\250710-RAB7A-col7nodrug-col8drug-go_20250710_154111\AssayPlate_Cellvis_P96-1.5H-N\No_Drug\First_Acquisition\F2\D07\out"

nuc_mask_path = os.path.join(out_dir, f"{nuc_id}_nuc_masks.tif")

trk_path = os.path.join(out_dir, f"{nuc_id}_joined_track.csv")
intermediate_trk_path = os.path.join(out_dir, f"{nuc_id}_intermediate_trk.csv")

adj_path = os.path.join(out_dir, f"{nuc_id}_padding_adjustments.csv")

# fig_dir = os.path.join(mv_parent_path,rf"figs\{field_of_interest}")

# if not os.path.exists(fig_dir):
#     os.makedirs(fig_dir)

#%%
# For use with manual segmentation of bursts
import random

out_list = os.listdir(out_dir)
random.choice(out_list)

#%%

mv = io.imread(nuc_mask_path)

trk = pd.read_csv(trk_path,index_col=0)
trk = trk.sort_values(by='frame').reset_index(drop=True)
trk['adjusted_frame'] = trk['frame'] - start

adj_df = pd.read_csv(adj_path,index_col = 0)
adj_df = adj_df.rename(columns={'timepoint':'frame'})

trk_2 = trk.merge(adj_df,on='frame')
trk_2['adjusted_y'] = trk_2['y'] + trk_2['adj_y']
trk_2['adjusted_x'] = trk_2['x'] + trk_2['adj_x']

trk_2['adjusted_rand_y'] = trk_2['rand_y'] + trk_2['adj_y']
trk_2['adjusted_rand_x'] = trk_2['rand_x'] + trk_2['adj_x']


trk_2 = trk_2.drop(columns=['track_id','state'])
trk_3 = trk_2.drop(columns=['start','end','len']).dropna()

intermediate_trk = pd.read_csv(intermediate_trk_path,index_col=0)

#%%

viewer = napari.Viewer()
# viewer.add_image(mv)

# labels_prev=np.array(nuc_masks_prev)
# viewer.add_image(labels_prev, name='prev_nucleus_labels',colormap ='b',opacity=0.5)

# labels=np.array(nuc_masks)
# viewer.add_image(labels, name='nucleus_labels',colormap ='r',opacity=0.5)
viewer.add_image(mv, name='segmented nucleus')

# Show tracking locus
viewer.add_points(
    trk_2[["adjusted_frame", "adjusted_y", "adjusted_x"]], size=8, edge_color="yellow", face_color="#ffffff00",name='spots'
)

# Show bursts
viewer.add_points(
    trk_3[["adjusted_frame", "adjusted_y", "adjusted_x"]], size=10, edge_color="red", face_color="#ffffff00",name='bursts'
)

# Show randomised points
viewer.add_points(
    trk_2[["adjusted_frame", "adjusted_rand_y", "adjusted_rand_x"]], size=8, edge_color="green", face_color="#ffffff00",name='random_position'
)


# Uncomment if looking at manually segmented bursts

# viewer.add_points(
#     manual_trk[["axis-0", "axis-1", "axis-2"]], size=5, edge_color="green", face_color="#ffffff00",name='manual'
# )


# viewer.add_points(
#     trk_spot_1[["frame", "y", "x"]], size=8, edge_color="red", face_color="#ffffff00",name='spots_1'
# )

# viewer.add_tracks(log_spots[["track_id", "frame", "y", "x"]], name='bursts',tail_width = 10 ,tail_length=25)


#%%
# Mark out areas of bursts
track_start_end = trk_3.groupby('filled_track_id')['adjusted_frame'].agg(['min','max']).rename(columns={'min':'start','max':'end'})
shade_regions = [(start,end) for start,end in zip(track_start_end['start'],track_start_end['end'])]

#%%
# # Generate raw trace

# ymin=min(trk['rand_masked_sum'])-max(trk['masked_sum'])*0.1
# ymax=max(trk['masked_sum'])+max(trk['masked_sum'])*0.1

# # Create the plot
# fig, ax1 = plt.subplots(figsize=(80, 8))

# # Plot 'masked_sum' and 'bgd_masked_sum' on the same y-axis (ax1)
# ax1.plot(trk['frame'], trk['masked_sum'], 'g-', label='Masked Sum')
# ax1.plot(trk['frame'], trk['rand_masked_sum'], 'r-', label='bgd_masked_sum')
# ax1.plot(trk['frame'], trk['mean_masked_sum'], 'r--', label='bgd_masked_sum')

# ax1.set_xlabel('Frame')
# ax1.set_ylabel('Spot total intensity', color='g')
# ax1.tick_params(axis='y', labelcolor='g')

# ax1.set_ylim([ymin,ymax])
# ax1.set_xlim([0,615])

# # Get the y-limits of the plot
# y_min, y_max = ax1.get_ylim()

# for start, end in shade_regions:
#     ax1.fill_betweenx([y_min, y_max], start, end, color='gray', alpha=0.3)
# ax1.hlines(y=0,xmin=min(trk['frame']),xmax=max(trk['frame']),color='k',alpha=0.5)

# # Show the plot
# # plt.savefig(os.path.join(fig_dir, f'{nuc_id}_raw_trace.png'), format='png', dpi=300)
# plt.show()

# del ymin,ymax,y_min,y_max,start,end,ax1,fig

#%%
# Generate background subtracted trace

# Create the plot
# fig, ax1 = plt.subplots(figsize=(16, 8))
fig, ax1 = plt.subplots(figsize=(16, 8))

ymin=min(trk['bgd_subtracted_rand_masked_sum'])-max(trk['bgd_subtracted_masked_sum'])*0.1
ymax=max(trk['bgd_subtracted_masked_sum'])+max(trk['bgd_subtracted_masked_sum'])*0.1

# Plot 'masked_sum' and 'bgd_masked_sum' on the same y-axis (ax1)
ax1.plot(trk['adjusted_frame'], trk['bgd_subtracted_masked_sum'], 'k-',alpha=0.4)
ax1.plot(trk['adjusted_frame'], trk['smoothened_bgd_subtracted_masked_sum'], 'k--', label='Locus smoothened')

# Show random or interpolated point
ax1.plot(trk['adjusted_frame'], trk['bgd_subtracted_rand_masked_sum'], 'r-',alpha=0.4)
ax1.plot(trk['adjusted_frame'], trk['smoothened_bgd_subtracted_rand_masked_sum'], 'r--', label='Random smoothened')

# Show the early track right after selecting for longest one
# ax1.scatter(x=intermediate_trk['frame'],y=intermediate_trk['bgd_subtracted_masked_sum'])

ax1.set_ylim([ymin,ymax])
ax1.set_xlim([0,trk['adjusted_frame'].max()])

# Get the y-limits of the plot
y_min, y_max = ax1.get_ylim()

for start, end in shade_regions:
    ax1.fill_betweenx([y_min, y_max], start, end, color='gray', alpha=0.3)
    
ax1.hlines(y=0,xmin=min(trk['adjusted_frame']),xmax=max(trk['adjusted_frame']),color='k',alpha=0.5)
plt.xlabel('Frame')
plt.ylabel('Bgd subtrcated total intensity (a.u.)')
plt.xticks(np.arange(max(trk['frame']),step=25))
plt.yticks()
plt.legend(bbox_to_anchor=[0.29,1.2],frameon=False,fontsize=18)

plt.xlim([0,None])

# Show the plot
sns.despine()
# plt.savefig(os.path.join(fig_dir, f'{nuc_id}_bgd_subtracted_trace.png'), format='png', dpi=300,bbox_inches='tight',transparent=False)
# plt.savefig(os.path.join(fig_dir, f'{nuc_id}_bgd_subtracted_trace.svg'), format='svg', dpi=300,bbox_inches='tight')

plt.show()

del ymin,ymax,y_min,y_max,start,end,ax1,fig

 #%%

# sys.path.insert(0, r"G:\Other computers\My Laptop\lab\experiments\ms2_crispr_endogenous_tagging\my_code")
# import spot_detection_functions as f

# # Generate background subtracted trace

# trk['mean_cdt1_intensity'] = trk['total_cdt1_intensity'] / trk['nuc_area']

# trk_mean_cdt1_int = trk['mean_cdt1_intensity'].to_numpy()
# trk_total_cdt1_int = trk['total_cdt1_intensity'].to_numpy()

# trk['smoothened_mean_cdt1_intensity'] = f.savitzky_golay(trk_mean_cdt1_int, window_size=7, order=0)
# trk['smoothened_total_cdt1_intensity'] = f.savitzky_golay(trk_total_cdt1_int, window_size=7, order=0)

    #%%

# Create the plot
# fig, ax1 = plt.subplots(figsize=(16, 8))
fig, ax1 = plt.subplots(figsize=(16, 8))

ymin=min(trk['total_cdt1_intensity'])-max(trk['total_cdt1_intensity'])*0.1
ymax=max(trk['total_cdt1_intensity'])+max(trk['total_cdt1_intensity'])*0.1

ax1.plot(trk['frame'], trk['total_cdt1_intensity'], 'k-',alpha=0.4)
ax1.plot(trk['frame'], trk['smoothened_total_cdt1_intensity'], 'k--', label='Locus smoothened')



# ymin=min(trk['smoothened_mean_cdt1_intensity'])-max(trk['smoothened_mean_cdt1_intensity'])*0.1
# ymax=max(trk['smoothened_mean_cdt1_intensity'])+max(trk['smoothened_mean_cdt1_intensity'])*0.1

# ax1.plot(trk['frame'], trk['mean_cdt1_intensity'], 'r-',alpha=0.4)
# ax1.plot(trk['frame'], trk['smoothened_mean_cdt1_intensity'], 'r--', label='Random smoothened')




# Show the early track right after selecting for longest one
# ax1.scatter(x=intermediate_trk['frame'],y=intermediate_trk['bgd_subtracted_masked_sum'])

ax1.set_ylim([ymin,ymax])
ax1.set_xlim([0,trk['frame'].max()])

# Get the y-limits of the plot
y_min, y_max = ax1.get_ylim()

for start, end in shade_regions:
    ax1.fill_betweenx([y_min, y_max], start, end, color='gray', alpha=0.3)
    
ax1.hlines(y=0,xmin=min(trk['frame']),xmax=max(trk['frame']),color='k',alpha=0.5)
plt.xlabel('Frame')
plt.ylabel('Total Cdt1 intensity (a.u.)')
plt.xticks(np.arange(max(trk['frame']),step=25))
plt.yticks()
# plt.legend(bbox_to_anchor=[0.29,1.2],frameon=False,fontsize=18)

plt.xlim([0,None])

# Show the plot
sns.despine()
# plt.savefig(os.path.join(fig_dir, f'{nuc_id}_bgd_subtracted_trace.png'), format='png', dpi=300,bbox_inches='tight')
# plt.savefig(os.path.join(fig_dir, f'{nuc_id}_bgd_subtracted_trace.svg'), format='svg', dpi=300,bbox_inches='tight')

plt.show()

del ymin,ymax,y_min,y_max,start,end,ax1,fig

#%%
# import tifffile as tiff

# for frame,img in enumerate(mv):
#     tiff.imwrite(os.path.join(fig_dir, f"{frame}_label.tif"), img, resolution=(1/0.216, 1/0.216), metadata ={'unit':'um'},imagej=True)

import imageio
import time

# Define the output video file
output_file = f'{nuc_id}_visualization.mp4'

writer = imageio.get_writer(os.path.join(fig_dir,output_file))

# Iterate over the frames in the image data
for i in range(mv.shape[0]):
    # Update the viewer to the current frame
    viewer.dims.current_step = (i, 0, 0)

    # Render the current frame
    screenshot = viewer.screenshot()

    # Write the frame to the video
    writer.append_data(screenshot)

    # Optional: Pause briefly to ensure rendering is complete
    time.sleep(0.1)

# Close the video writer
writer.close()

print(f"Video saved as {output_file}")
