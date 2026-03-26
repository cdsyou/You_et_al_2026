# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:28:47 2024

@author: cdsyou
"""

#%%

import os
import glob
import pandas as pd
import re
from sklearn import metrics
from tqdm import tqdm

import sys

sys.path.insert(0, r"INSERT CODE LOCATION")
import spot_detection_functions as f

#%%
def get_auc(df,frame_rate):
    # Group by 'group' column
    grouped = df.groupby('filled_track_id')
    
    # Initialize empty dictionaries to store the results
    grouped_frame_dict = {}
    grouped_bgd_sub_masked_sum_dict = {}
    
    # Extract col1 and col2 for each group
    for name, group in grouped:
        grouped_frame_dict[name] = group['frame'].reset_index(drop=True)
        grouped_bgd_sub_masked_sum_dict[name] = group['smoothened_bgd_subtracted_masked_sum'].reset_index(drop=True)

    # Convert dictionaries to lists of lists
    frame_lists = [list(values*frame_rate) for values in grouped_frame_dict.values()]
    bgd_masked_sum_lists = [list(values) for values in grouped_bgd_sub_masked_sum_dict.values()]    
    
    auc_l = [metrics.auc(xx, yy) for xx,yy in zip(frame_lists,bgd_masked_sum_lists)]

    return auc_l

def fill_off_periods(df):
    nan_mask = df['filled_track_id'].isna()

    # Use diff to identify start and end of nan periods
    nan_periods = (nan_mask != nan_mask.shift()).cumsum() * nan_mask

    # Get unique periods of nan (excluding 0 which means non-nan periods)
    unique_nan_periods = nan_periods[nan_periods != 0].unique()

    # Replace periods of nan with descending integer labels
    for i, period in enumerate(sorted(unique_nan_periods, reverse=True), start=1):
        df.loc[nan_periods == period, 'filled_track_id'] = -i

    # Convert the 'values' column back to the original type
    df.loc[:,'filled_track_id'] = df.loc[:,'filled_track_id'].astype(int)
    
    return df

def remove_first_and_last_blocks(df, column_name):
    # Identify the first block
    first_block_value = df[column_name].iloc[0]
    second_block_start = df[df[column_name] != first_block_value].index[0]

    # Identify the last block
    last_block_value = df[column_name].iloc[-1]
    last_block_start = df[df[column_name] != last_block_value].index[-1] + 1

    # Remove the first and last blocks
    new_df = df.iloc[second_block_start:last_block_start]
    
    return new_df

def process_periods(trk,trk_path,cond,field,frame_rate,pix_x,pix_y):
    
    name = trk_path.split('\\')[-1].replace('_joined_track.csv','')

    # Remove the first and last instance of ON or OFF since we cannot determine when they begin or end
    trk_2 = remove_first_and_last_blocks(trk,'filled_state')
        
    # Only keep tracks that are more than 2 frames long
    trk_3 = f.replace_short_on(trk_2,threshold=3)
                
    # Fill off period track ids
    trk_3 = fill_off_periods(trk_3)
    
    on_trk = trk_3[trk_3['filled_state'] == 1]
    off_trk = trk_3[trk_3['filled_state'] == 0]
    
    on_trk_len = on_trk.groupby('filled_track_id')['frame'].count().reset_index().rename(columns={'frame':'length'})
    on_trk_avg_area = on_trk.groupby('filled_track_id')['nuc_area'].mean().reset_index().rename(columns={'nuc_area':'period_avg_nuc_area'})
    on_trk_area_diff = on_trk.groupby('filled_track_id')['nuc_area'].agg(['first', 'last'])
    on_trk_area_diff['area_diff'] = (on_trk_area_diff['last'] - on_trk_area_diff['first'])*pix_x*pix_y
    
    off_trk_len = off_trk.groupby('filled_track_id')['frame'].count().reset_index().rename(columns={'frame':'length'})
    off_trk_avg_area = off_trk.groupby('filled_track_id')['nuc_area'].mean().reset_index().rename(columns={'nuc_area':'period_avg_nuc_area'})
    off_trk_area_diff = off_trk.groupby('filled_track_id')['nuc_area'].agg(['first', 'last'])
    off_trk_area_diff['area_diff'] = (off_trk_area_diff['last'] - off_trk_area_diff['first'])*pix_x*pix_y

    on_auc_l = get_auc(on_trk,frame_rate)
        
    on_df = on_trk_len.merge(on_trk_avg_area,on='filled_track_id')
    on_df = on_df.merge(on_trk_area_diff,on='filled_track_id')

    off_df = off_trk_len.merge(off_trk_avg_area,on='filled_track_id')
    off_df = off_df.merge(off_trk_area_diff,on='filled_track_id')

    on_df['cond'] = cond
    off_df['cond'] = cond
    
    on_df['id'] = trk_path.split('\\')[-2] + '_' + name
    off_df['id'] = trk_path.split('\\')[-2] + '_' + name
    
    on_df['state'] = 1
    off_df['state'] = 0
    
    on_df['field'] = int(field)
    off_df['field'] = int(field)
    
    on_df['length_min'] = on_df['length']*frame_rate/60
    off_df['length_min'] = off_df['length']*frame_rate/60

    on_df['period_avg_nuc_area_um2'] = on_df['period_avg_nuc_area']*pix_x*pix_y
    off_df['period_avg_nuc_area_um2'] = off_df['period_avg_nuc_area']*pix_x*pix_y

    on_df['auc'] = on_auc_l
    on_df['auc_area_um2'] = on_df['auc'] / on_df['period_avg_nuc_area_um2']
    
    on_df['auc_area_um2_length_min'] = on_df['auc_area_um2'] / on_df['length_min']
    
    # Rearrange dataframe
    on_df = on_df[['id','field','filled_track_id','cond','state','area_diff','period_avg_nuc_area','period_avg_nuc_area_um2','length','length_min','auc', 'auc_area_um2','auc_area_um2_length_min']]
    off_df = off_df[['id','field','filled_track_id','cond','state','area_diff','period_avg_nuc_area','period_avg_nuc_area_um2','length','length_min']]
    
    return on_df, off_df

#%%
outs = r"SOURCE DIRECTORY FOR OUTPUT FILES FROM SEGMENTATION AND PRELIMINARY ANALYSIS"

df_out_dir = r"OUTPUT DIRECTORY"
fig_dir = r"FIGURE DIRECTORY"

# pixel dimension in um 
pix_x = 0.216
pix_y = 0.216

dmso_code = 5

# Seconds between frames
frame_rate = 100

if not os.path.exists(df_out_dir):
    os.makedirs(df_out_dir)

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

file_ending = 'joined_track.csv'

#%%
# Define the regex pattern as a raw string
cond_pattern = 'col(.)_row3_field(.*)_out'

#%%

on_df_l=[]
off_df_l=[]
flag_l=[]
flag_2_l=[]

for out_dir in tqdm(os.listdir(outs)):
    if out_dir.endswith('_out'):
        search_pattern = os.path.join(outs, out_dir, f'*_{file_ending}')
    
        # Use glob to find all files matching the pattern
        files = glob.glob(search_pattern, recursive=True)
        
        cond_match = re.search(cond_pattern, out_dir)
        
        if int(cond_match.group(1)) == dmso_code:
            cond = 'dmso'
        else:
            ValueError('match not found')
        
        field = cond_match.group(2)
    
        for trk_path in files:
                        
            trk = pd.read_csv(trk_path,index_col=0)
                
            if len(trk['filled_track_id'].dropna().unique()) < 2:
                continue
            
            on_df, off_df = process_periods(trk,trk_path,cond,field,frame_rate,pix_x,pix_y)
            on_df_l.append(on_df)
            off_df_l.append(off_df)


#%%

on = pd.concat(on_df_l)
off = pd.concat(off_df_l)

on.to_csv(os.path.join(df_out_dir,'on.csv'))
off.to_csv(os.path.join(df_out_dir,'off.csv'))

