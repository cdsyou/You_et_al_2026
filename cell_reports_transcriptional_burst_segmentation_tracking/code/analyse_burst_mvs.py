# -*- coding: utf-8 -*-
"""
Created on Wed May 29 21:45:17 2024

@author: cdsyou
"""

from laptrack import LapTrack
import sys

sys.path.insert(0, r"INSERT CODE LOCATION")
import spot_detection_functions as f

#%%

def analyse_mv(frame_l,masked_img_l,masked_img_norm_l,nuc_area_l,cond,frame_tolerance = 3,distance_tolerance = 5):
    
    # max_sigma value should be such that the disk drawn around the potential burst is not fully saturated 
    if cond == 'dmso':
        max_sigma = 2
        sigma_2 = 2
        
        
    # Segment nucleus and find spots through LoG
    spots_df,spots_bgd_df,valid_mask_l = f.build_spots_df(frame_l,masked_img_l,masked_img_norm_l,nuc_area_l,max_sigma,sigma_2)

    # Filter based on background (ie erode)
    spots_df_compare = spots_df.merge(spots_bgd_df,on='frame')
    spots_df_2 = spots_df[spots_df_compare['bgd_subtracted_masked_sum'] > spots_df_compare['bgd_subtracted_rand_masked_std']*1.96].reset_index(drop=True)

    # Spots should have neighbours in time and space
    spots_df_3 = f.find_spots_w_neighbour(spots_df_2,near_spots_rqd=2, frame_tolerance=frame_tolerance,distance_tolerance=distance_tolerance).reset_index(drop=True)
    
    # If too few spots detected, move on
    if len(spots_df_3) < 1:
        ok = False
        error=1
        track_df_10 = False
        track_df_3 = False
        
        return ok,error,track_df_10,track_df_3

    # Dilate
    spots_df_4 = f.dilate_spots(spots_df_2,spots_df_3,frame_tolerance=frame_tolerance,distance_tolerance=distance_tolerance)
    
    # If too few spots detected, move on
    if len(spots_df_4) < 2:
        ok = False
        error=2
        track_df_10 = False
        track_df_3 = False

        return ok,error,track_df_10,track_df_3
    
    # Construct tracks
    lt = LapTrack(track_cost_cutoff=distance_tolerance**2, gap_closing_cost_cutoff=distance_tolerance**1, gap_closing_max_frame_count=frame_tolerance)
    track_df, _, _ = lt.predict_dataframe(spots_df_4, ["y", "x"], only_coordinate_cols=False)
    track_df = track_df.reset_index()
    track_df=track_df.drop(columns=['frame','index','tree_id'])
    track_df=track_df.rename(columns={'frame_y':'frame'})
    
    # Only keep bursts longer than 2 frame length (ie bursts have to last at least 5 min in length)
    trk_len = track_df.groupby('track_id')['frame'].agg(['min', 'max']).reset_index()
    trk_len = trk_len.rename(columns={'min':'start','max':'end'})
    trk_len['len'] = trk_len['end'] - trk_len['start']+1
    trk_len['keep'] = trk_len['len'] > 2
    track_df = track_df.merge(trk_len,on='track_id')
    track_df_2 = track_df[track_df['keep'] == True].drop(columns=['keep'])
    
    # If too few tracks detected, move on
    if len(track_df_2['track_id'].dropna().unique()) < 2:
        ok = False
        error=3
        track_df_10 = False
        track_df_3 = False
    
        return ok,error,track_df_10,track_df_3
    
    # If tracks are overlapping, only keep the longest overlapping track
    # Could instead join tracks based on distance
    track_df_3 = f.keep_long_trks(track_df_2)
    
    track_df_4 = f.fill_missing_frames(track_df_3, frame_l,valid_mask_l=valid_mask_l)
    
    track_df_4 = track_df_4.drop(columns=['sigma', 'sigma_2', 'nuc_area', 'masked_sum',
                                            'bgd_subtracted_masked_sum'])
    
    # Fill in missing track id and states between frames in the same track
    track_df_5 = f.fill_trk_ids_states(track_df_4)  
    track_df_5['nuc_area'] = nuc_area_l
    
    track_df_5 = track_df_5.merge(spots_bgd_df,on='frame')
    
    # Get the spot sums of filled in and existing spots
    track_df_6 = f.recalc_spot_sums(track_df_5,masked_img_l,sigma_2=sigma_2)
    
    # Smoothen the masked sums and nuclear area
    track_df_6_nuc_raw = track_df_6['nuc_area'].to_numpy()
    
    track_df_6_raw = track_df_6['bgd_subtracted_masked_sum'].to_numpy()
    track_df_6_rand_raw = track_df_6['bgd_subtracted_rand_masked_sum'].to_numpy()
    
    track_df_6['smoothened_nuc_area'] = f.savitzky_golay(track_df_6_nuc_raw, window_size=11, order=0)
    
    track_df_6['smoothened_bgd_subtracted_masked_sum'] = f.savitzky_golay(track_df_6_raw, window_size=5, order=0)
    track_df_6['smoothened_bgd_subtracted_rand_masked_sum'] = f.savitzky_golay(track_df_6_rand_raw, window_size=5, order=0)
    
    # Erode tracks based on standard deviation of bgd subtracted masked sum
    track_df_7 = f.erode_tracks(track_df_6,threshold_adj=2.58)
    
    # If too few tracks detected, move on
    if len(track_df_7['filled_track_id'].dropna().unique()) < 2:
        ok = False
        error=4
        track_df_10 = False
        track_df_3 = False
    
        return ok,error,track_df_10,track_df_3
    
    # Dilate tracks again based on standard deviation threshold
    track_df_8 = f.dilate_tracks(track_df_7,threshold_adj=1.96)
    
    # Only keep tracks that are more than 2 frames long
    track_df_9 = f.replace_short_on(track_df_8,threshold=3)
    
    # Merge any segments that are close together in time (no assumptions made about space since assumed to be single bursting allele)
    track_df_10 = f.merge_close_tracks(track_df_9,closed_gap_length=1)    
    
    track_df_10 = track_df_10.sort_values(by='frame').reset_index(drop=True)
    
    ok = True
    error=0
    
    return ok,error,track_df_10,track_df_3
