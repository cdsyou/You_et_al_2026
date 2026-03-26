# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:05:46 2024

@author: cdsyou
"""

import numpy as np
from skimage import io
import os
import sys
import tifffile as tiff
from stardist.models import StarDist2D
from laptrack import LapTrack
from skimage.measure import regionprops_table
import pandas as pd
from csbdeep.utils import normalize
# import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import re
from datetime import datetime

sys.path.insert(0, r"INSERT CODE LOCATION")

import analyse_burst_mvs as a

model = StarDist2D.from_pretrained('2D_versatile_fluo')

#%%
# Variables
frame_tolerance = 3
distance_tolerance = 4
crop_size = 200
pix_x = 0.216
pix_y = 0.216
trk_len_threshold = 50
min_area = 1000
max_area = 5000
trk_cost=25

#%%

source_directory = r"IMAGE SOURCE DIRECTORY"

source_mv_dir = source_directory.split('\\')[1]
out_mv_dir = source_mv_dir + '_mvs'

# Directory to contain organised movies and analysed data
mv_parent_dir = r"MOVIE PARENT DIRECTORY"
out_parent_dir = r"OUTPUT PARENT DIRECTORY"
labels_dir = r"LABELS OUTPUT DIRECTORY"

if not os.path.exists(mv_parent_dir):
    os.makedirs(mv_parent_dir)

if not os.path.exists(out_parent_dir):
    os.makedirs(out_parent_dir)
    
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

#%%

def copy_files_with_pattern(source_dir, destination_dir, file_pattern):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        
    for file_name in os.listdir(source_dir):
        
        try: 
            type(re.search(file_pattern, file_name)[0]) == str
            
        except:
            pass  
        
        else:
            # Construct source and destination paths
            source_path = os.path.join(source_dir, file_name)
            destination_path = os.path.join(destination_dir, file_name)
            # Copy the file
            shutil.copy(source_path, destination_path)

#%%
# Move and organise files for analysis

column = 9
field_n = 12

field_l = np.arange(1,field_n+1)

field_l_2 = []

for field in field_l:
    str_field = str(field)
    if len(str_field) == 1:
        field_new = '0' + str_field
        field_l_2.append(field_new)
    else:
        field_new = str_field
        field_l_2.append(field_new)

col_l = [column for i in field_l_2]

for col,field in zip(col_l,field_l_2):
    
    destination_directory = os.path.join(mv_parent_dir, f"col{col}_row3_field{field}")
    file_pattern = rf'AssayPlate_Cellvis_P96-1.5H-N_D0{col}_T[0-9]*F0{field}.*C01.tif'

    copy_files_with_pattern(source_directory, destination_directory, file_pattern)

#%%
pattern = "col(.)_row3_field(.*)"

try:
    error_l_l = []
    
    for mv_dir in tqdm(os.listdir(mv_parent_dir)):
        
        name_match = re.search(pattern, mv_dir)
        
        if int(name_match.group(1)) == 5:
            cond = 'dmso'
        else:
            ValueError('match not found')
            
        col = name_match.group(1)
        field = name_match.group(2)
        
        out_dir = os.path.join(out_parent_dir,f'col{col}_row3_field{field}_out')
        label_out_dir = os.path.join(labels_dir,f'col{col}_row3_field{field}_label_out')
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        if not os.path.exists(label_out_dir):
            os.makedirs(label_out_dir)
                
        mv_input_dir = os.path.join(mv_parent_dir,mv_dir)
        
        # Get a list of all .tif files in the directory
        tif_files = [f for f in sorted(os.listdir(mv_input_dir)) if f.endswith('.tif')]
        
        # Initialize an empty list to hold the images
        image_stack = []
        
        # Loop through the files and read them into the list
        for file_name in tif_files:
            image_path = os.path.join(mv_input_dir, file_name)
            img = io.imread(image_path)  # or tifffile.imread(image_path)
            image_stack.append(img)
        
        # Convert the list to a 3D numpy array (movie)
        mv = np.stack(image_stack, axis=0)
        
        # Initialize a list to store segmented nuclei for each frame
        regionprops_l = []
        segmented_images = []
        
        current_time = datetime.now().ctime()
        print(f'At {current_time}: segmenting nuclei in {mv_dir}')
        
        # Segment each frame in the movie
        for frame, img in enumerate(mv):
            image_norm = normalize(img, 1,99.8, axis = (0,1))
            # Predict the nuclei segmentation for the current frame
            labels, _ = model.predict_instances(image_norm)
            
            # Change to uint16 to enable saving at the end
            labels = labels.astype(np.uint16)
            
            # Append the segmented nuclei to the list
            segmented_images.append(labels)
            
            df = pd.DataFrame(regionprops_table(labels, properties=["label", "centroid",'area','bbox']))
            df["frame"] = frame
            regionprops_l.append(df)
            
        current_time = datetime.now().ctime()
        print(f'At {current_time}: finished segmenting {mv_dir}')
        
        # Convert the list of segmented images into a 3D array
        segmented_movie = np.stack(segmented_images, axis=0)
        
        regionprops_df = pd.concat(regionprops_l)
        
        # fig,ax=plt.subplots()
        # plt.hist(data=regionprops_df, x='area',bins=100)
        # plt.xlim([0,max_area + 500])
        # plt.show()
        
        height,width = img.shape
        
        # Remove segmented nuclei that are either too small or too large
        regionprops_df = regionprops_df[(regionprops_df['area'] > min_area) & (regionprops_df['area'] < max_area)]
        
        # Remove segmented nuclei on the frame border
        regionprops_df = regionprops_df[(regionprops_df['bbox-0'] > 0) & (regionprops_df['bbox-1'] > 0) & 
                                            (regionprops_df['bbox-2'] < height) & (regionprops_df['bbox-3'] < width)]
        
        # Track the nuclei segmented - Note that no gaps are allowed 
        # This is to allow easier downstream analysis - to enable gaps, must allow for additional segmentation
        lt = LapTrack(track_cost_cutoff=trk_cost**2)
        
        track_df, _, _ = lt.predict_dataframe(
            regionprops_df.copy(),
            coordinate_cols=["centroid-0", "centroid-1"],
            only_coordinate_cols=False,
        )
        
        current_time = datetime.now().ctime()
        print(f'At {current_time}: finished tracking {mv_dir}')
        
        track_df = track_df.reset_index()[['frame','label','track_id']]
        regionprops_df = regionprops_df.merge(track_df,on=['frame','label'])
        
        # Only keep nuclei with tracks longer than certain length
        
        regionprops_df = regionprops_df.groupby('track_id').filter(lambda x: len(x) > trk_len_threshold)
        
        filtered_labels_l = []
        
        # Keep only labels that have passed the filtering process
        for frame,label_img in enumerate(segmented_movie):
            desired_ids = np.unique(regionprops_df[regionprops_df['frame'] == frame]['label'])
            filtered_labels = np.isin(label_img, desired_ids) * label_img
            filtered_labels_l.append(filtered_labels)
        
        label_mv = np.stack(filtered_labels_l, axis=0)    
        
        error_l = []
        track_l = regionprops_df['track_id'].unique()
        
        current_time = datetime.now().ctime()
        print(f'At {current_time}: segmenting and tracking bursts in {mv_dir}')

        for track in track_l:
            
            temp_df = regionprops_df[regionprops_df['track_id'] == track]
            
            centroid = 'y' + str(round(list(temp_df['centroid-0'])[0])) + '_x' +  str(round(list(temp_df['centroid-1'])[0]))
        
            frame_l = list(temp_df['frame'])
            label_l = list(temp_df['label'])
            nuc_area_l = list(temp_df['area'])
            y_min_l = list(temp_df['bbox-0'])
            y_max_l = list(temp_df['bbox-2'])
            x_min_l = list(temp_df['bbox-1'])
            x_max_l = list(temp_df['bbox-3'])
            
            # Nuclear area should not fluctuate by more than 25% in a single time point
            percent_change = [(nuc_area_l[i] - nuc_area_l[i-1]) / nuc_area_l[i-1] * 100 for i in range(1, len(nuc_area_l))]
            
            # Not a pretty solution - throws out a lot of nuclei
            if abs(max(percent_change,key=abs)) > 25:
                error = 7
                error_l.append(error)
                continue
            
            start = frame_l[0]
            end = frame_l[-1]
            
            mv_len = len(frame_l)
        
            masked_img_l = []
            masked_img_norm_l = []
            padded_masked_img_l = []
            crop_adj_df = pd.DataFrame(columns=['frame',"adj_y", "adj_x"])
            adj_y_l = []
            adj_x_l = []
        
            # Check that the nuclei tracking is longer than threshold and that it is continuous throughout it
            if mv_len == len(np.arange(start,end+1)) and (end - start > trk_len_threshold):
                
                # Iterate through the frame and collect the masked images and masked normalised images
                for frame,label,y_min,y_max,x_min,x_max in zip(frame_l,label_l,y_min_l,y_max_l,x_min_l,x_max_l):
                    
                    cropped_img = mv[frame][y_min:y_max, x_min:x_max]
                    cropped_label = label_mv[frame][y_min:y_max, x_min:x_max]
                    
                    mask = cropped_label == label
                    masked_img = np.zeros_like(cropped_img)
                    masked_img[mask] = cropped_img[mask]
                                        
                    masked_img_norm = normalize(masked_img, 1,99.8, axis = (0,1))
                    
                    masked_img_l.append(masked_img)
                    masked_img_norm_l.append(masked_img_norm)
                    
                    # Create padded movie of segmented nuclei tracking
                    height,width = masked_img.shape
                    padded_masked_img = np.zeros((crop_size, crop_size), dtype=masked_img.dtype)
                    
                    crop_start_y = round((crop_size - height) /2)
                    crop_start_x = round((crop_size - width) /2)
                    crop_end_y = height + round((crop_size - height) /2)
                    crop_end_x = width + round((crop_size - width) /2)
                    
                    adj_y_l.append(crop_start_y)
                    adj_x_l.append(crop_start_x)
                    
                    padded_masked_img[crop_start_y:crop_end_y, crop_start_x:crop_end_x] = masked_img[:,:]
                    padded_masked_img_l.append(padded_masked_img)
                            
                # Do the burst tracking
                ok,error,trk,intermediate_trk = a.analyse_mv(frame_l,masked_img_l,masked_img_norm_l,nuc_area_l,cond,
                                                             frame_tolerance = frame_tolerance, distance_tolerance = distance_tolerance)
                error_l.append(error)
                
                if ok == True:
                
                    trk.to_csv(os.path.join(out_dir, f'frame{start}_{centroid}_track{track}_len{mv_len}_joined_track.csv'))
                    intermediate_trk.to_csv(os.path.join(out_dir, f'frame{start}_{centroid}_track{track}_len{mv_len}_intermediate_trk.csv'))
                    
                    padded_masked_mv = np.stack(padded_masked_img_l, axis=0)
            
                    tiff.imwrite(os.path.join(out_dir, f"frame{start}_{centroid}_track{track}_len{mv_len}_nuc_masks.tif"), 
                                 padded_masked_mv, resolution=(1/pix_x, 1/pix_y), metadata ={'unit':'um'},imagej=True)
                    
                    crop_adj_df["frame"] = frame_l
                    crop_adj_df["adj_y"] = adj_y_l
                    crop_adj_df["adj_x"] = adj_x_l
                    
                    crop_adj_df.to_csv(os.path.join(out_dir, f'frame{start}_{centroid}_track{track}_len{mv_len}_padding_adjustments.csv'))
                    
                    regionprops_df.to_csv(os.path.join(out_dir, f'frame{start}_{centroid}_track{track}_len{mv_len}_regionprops.csv'))
        
                else:
                    pass
        
            else:
                if mv_len != len(np.arange(start,end+1)):
                    error = 5
                    error_l.append(error)
                elif (end - start >= trk_len_threshold):
                    error = 6
                    error_l.append(error)
                else:
                    pass
        
        current_time = datetime.now().ctime()
        print(f'At {current_time}: finished segmenting and tracking bursts in {mv_dir}')

        for frame,label_img in enumerate(label_mv):
            tiff.imwrite(os.path.join(label_out_dir, f"{frame}_label.tif"), label_img, resolution=(1/pix_x, 1/pix_y), metadata ={'unit':'um'},imagej=True)
        
        error_l_l.append(error_l)
        
except Exception as e:
    print(f"Failed at function3 with input {mv_input_dir}: {e}")

