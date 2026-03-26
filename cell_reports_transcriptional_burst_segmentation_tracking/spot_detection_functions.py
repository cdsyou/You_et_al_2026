# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:47:24 2024

@author: cdsyou
"""

import numpy as np
import pandas as pd
from skimage import feature,morphology,exposure,draw
from scipy.spatial.distance import euclidean
from skimage.measure import regionprops
import matplotlib.pyplot as plt

#%%
def segment(image,model):
    '''

    Parameters
    ----------
    image : np array
        A single frame from tif movie of segmented nucleus.

    Returns
    -------
    masked_image : array
        original image that has been masked for nucleus.
    masked_norm_image : array
        rescaled image that has been masked for nucleus.
    nuc_area : int
        masked nuclear area.

    '''    
    # Normalize the image
    image_norm = exposure.rescale_intensity(image, out_range=(0, 1))
    
    # Segment the nucleus using pre-trained Stardist versatile 2D fluorescent model    
    labels, _ = model.predict_instances(image_norm)

    # Get image dimensions
    height, width = image.shape
    
    # Calculate the center of the image
    center_y, center_x = height / 2, width / 2
    
    # Set image boundary
    img_x_min = 0
    img_y_min = 0
    img_y_max = height
    img_x_max = width
    
    # Check if zero regions exist
    # If they exist, then we want to do the normalisation and nuclear segmentation only using non zero regions
    zero_rows = np.where(np.all(image == 0, axis=1))[0]
    zero_cols = np.where(np.all(image == 0, axis=0))[0]
    
    zero_region_exists = len(zero_rows) > 0 or len(zero_cols) > 0
    
    if zero_region_exists:
    
        # If zero regions exist, find their min and max extents
        zero_region_x_min = zero_cols.min() if len(zero_cols) > 0 else None
        zero_region_x_max = zero_cols.max() + 1 if len(zero_cols) > 0 else None
        zero_region_y_min = zero_rows.min() if len(zero_rows) > 0 else None
        zero_region_y_max = zero_rows.max() + 1 if len(zero_rows) > 0 else None    
        
        if (len(zero_rows) > 0) and (len(zero_cols) > 0):
            # There are zero strips on both horizontal and vertical sides
            # Based on the type of zero region, identify the image boundaries
            if zero_region_x_min == 0 and zero_region_y_min == 0:
                img_x_min = zero_region_x_max
                img_y_min = zero_region_y_max
            elif zero_region_x_min == 0 and zero_region_y_max == height:
                img_x_min = zero_region_x_max
                img_y_max = zero_region_y_min
            elif zero_region_x_max == width and zero_region_y_max == height:
                img_x_max = zero_region_x_min
                img_y_max = zero_region_y_min
            elif zero_region_x_max == width and zero_region_y_min == 0:
                img_x_max = zero_region_x_min
                img_y_min = zero_region_y_max
            else:
                pass
            
        else:
            # Based on the type of zero region, identify the image boundaries
            if zero_region_x_min == 0:
                img_x_min = zero_region_x_max
            elif zero_region_y_min == 0:
                img_y_min = zero_region_y_max
            elif zero_region_x_max == width:
                img_x_max = zero_region_x_min
            elif zero_region_y_max == height:
                img_y_max = zero_region_y_min
            else:
                pass
    
    else:
        pass
    
    # Find the segmented object closest to the center
    props = regionprops(labels)

    # Check if at least one object is segmented
    # If not, do not treat this frame as being valid for analysis
    if len(props) < 1:
        
        masked_image = None
        masked_norm_image = None
        nuc_area = None
        
        return masked_image,masked_norm_image,nuc_area
        
    # Function to calculate the distance from the centroid to the center of the image
    def distance_to_center(prop):
        centroid_y, centroid_x = prop.centroid
        return np.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2)
        
    # Find the object with the shortest distance to the center
    closest_obj = min(props, key=distance_to_center)
    closest_obj_label = closest_obj.label
    
    # Check if the object touches the boundary
    min_row, min_col, max_row, max_col = closest_obj.bbox
    
    touches_boundary = (
        min_row == img_y_min or
        min_col == img_x_min or
        max_row == img_y_max or
        max_col == img_x_max
    )
        
    # Check if nucleus intersects with the image boundary
    # If it does, do not treat this frame as being valid for analysis
    if touches_boundary:
        
        masked_image = None
        masked_norm_image = None
        nuc_area = None
        
        return masked_image,masked_norm_image,nuc_area

    # Create a mask for the largest object
    closest_obj_mask = (labels == closest_obj_label)
    
    # Step 8: Extract pixel values within the object mask from the normalised and original image
    masked_norm_image = np.zeros_like(image_norm)
    masked_norm_image[closest_obj_mask] = image_norm[closest_obj_mask]
    
    masked_image = np.zeros_like(image)
    masked_image[closest_obj_mask] = image[closest_obj_mask]
    
    # Step 9: Calculate the area of the largest object mask
    nuc_area = closest_obj_mask.sum()
    
    if nuc_area < 100:
        
        masked_image = None
        masked_norm_image = None
        nuc_area = None
        
        return masked_image,masked_norm_image,nuc_area

    return masked_image,masked_norm_image,nuc_area

#%%
def get_masked_sum(masked_image, y, x, sigma_2):
    '''

    Parameters
    ----------
    masked_image : array
        Image after masking for nucleus.
    y : int
        y coordinate of spot.
    x : int
        x coordinate of spot.
    sigma : float
        standard deviation of Gaussian kernel that detected blob.

    Returns
    -------
    masked_sum : int
        Function to create a circular mask and get the total values inside the mask.

    '''
    radius = int(np.sqrt(2) * sigma_2)
    mask = np.zeros(masked_image.shape, dtype=bool)
    rr, cc = draw.disk((y, x), radius, shape=masked_image.shape)
    mask[rr, cc] = True
    masked_sum = np.sum(masked_image[mask])
    
    return masked_sum

def bgd_msk_sum(masked_image,frame,sigma_2):
    
    radius = int(np.sqrt(2) * sigma_2)

    # Get spot area for the given sigma_2 value
    rr, cc = draw.disk((0, 0), radius)
    spot_area = len(rr)
    
    # Get the sum of the disk when pixels are mean values inside the nucleus
    bgd_df = pd.DataFrame(data={"frame":[frame]})
    
    # Step 1: Convert to binary mask
    binary_mask = masked_image > 0
    mean_nuc_pixel = np.mean(masked_image[binary_mask])
    
    # Create a square array large enough to contain the disk
    diameter = 2 * radius + 1
    ftprint = np.zeros((diameter, diameter), dtype=bool)
    
    # Get disk coordinates relative to the footprint center
    rr, cc = draw.disk((radius, radius), radius)
    
    # Set those pixels True in the footprint
    ftprint[rr, cc] = True
    
    # Now use footprint in erosion
    valid_mask = morphology.binary_erosion(binary_mask, footprint=ftprint)
            
    bgd_df['mean_nuclear_pixel'] = mean_nuc_pixel
    bgd_df['mean_bgd_sum'] = mean_nuc_pixel*spot_area
    
    return bgd_df,valid_mask

def get_random_disk_centers(masked_image, existing_centers, sigma_2):
    
    radius = int(np.sqrt(2) * sigma_2)

    # Step 1: Convert to binary mask
    binary_mask = masked_image > 0
    
    # Create a square array large enough to contain the disk
    diameter = 2 * radius + 1
    ftprint = np.zeros((diameter, diameter), dtype=bool)
    
    # Get disk coordinates relative to the footprint center
    rr, cc = draw.disk((radius, radius), radius)
    
    # Set those pixels True in the footprint
    ftprint[rr, cc] = True
    
    # Now use footprint in erosion
    valid_mask = morphology.binary_erosion(binary_mask, footprint=ftprint)
    
    if len(existing_centers) > 0:
    
        # Step 3: Create exclusion zones for existing disk centers
        exclusion_mask = np.zeros(valid_mask.shape, dtype=bool)
        for center in existing_centers:
            rr, cc = draw.disk(center, radius*2, shape=valid_mask.shape)
            exclusion_mask[rr, cc] = True
    
        # Step 4: Subtract exclusion zones from valid mask
        valid_mask = np.logical_and(valid_mask, ~exclusion_mask)
        
    else:
        pass
    
    # Step 5: Get the valid positions
    valid_positions = np.argwhere(valid_mask)
    
    if len(valid_positions) == 0:
        raise ValueError("No valid positions found. The object might be too small for the given radius.")
    
    draw_pool = np.arange(valid_positions.shape[0])
    # Step 6: Randomly select a valid position
    rand_positions = valid_positions[np.random.choice(draw_pool,round(len(draw_pool)*.33),replace=False)]
        
    return rand_positions

def find_spots_w_neighbour(df,near_spots_rqd=2, frame_tolerance=3,distance_tolerance=5):
    
    # Initialize a list to collect the filtered spots
    keep_indices = []
    
    # Iterate over each unique frame
    for frame in df['frame'].unique():
        spots_in_frame = df[df['frame'] == frame]
        for index, spot in spots_in_frame.iterrows():
            found_neighbor=0
    
            # Calculate the range of neighboring frames to consider
            start_frame = max(frame - frame_tolerance, df['frame'].min())
            end_frame = min(frame + frame_tolerance, df['frame'].max())
            
            for neighboring_frame in range(start_frame, end_frame + 1):
                if found_neighbor==near_spots_rqd:
                    break
                elif neighboring_frame == frame:
                    continue
                else:
                    neighboring_spots = df[df['frame'] == neighboring_frame]
                    if len(neighboring_spots) > 0:
                        time_diff = abs(neighboring_frame - frame)
                        time_scaled_dist_thresh = time_diff*distance_tolerance
                        for _, neighbor in neighboring_spots.iterrows():
                            is_within_distance = euclidean((spot['x'], spot['y']), (neighbor['x'], neighbor['y'])) <= time_scaled_dist_thresh
                            if is_within_distance:
                                found_neighbor += 1
                                
                                if found_neighbor==near_spots_rqd:
                                    keep_indices.append(index)
                                    break
                    else:
                        pass
    
    # Get the spots that have neighbouring spots in a given time frame_tolerance
    df_2 = df[df.index.isin(keep_indices)]
    
    return df_2

def dilate_spots(df,df2,frame_tolerance=3,distance_tolerance=3):
        
    # List to store results
    results = []
    
    # Iterate over df_3 to find matching spots in spots_df
    for _, row in df2.iterrows():
        frame = row['frame']
        x = row['x']
        y = row['y']
        
        # Filter spots_df to get spots within the frame tolerance
        df_filtered = df[(df['frame'] >= frame - frame_tolerance) & (df['frame'] <= frame + frame_tolerance)]
        
        # Check distance for each filtered spot
        for _, row2 in df_filtered.iterrows():
            frame_2 = row2['frame']
            frame_diff = abs(frame-frame_2)
            
            if euclidean([x, y], [row2['x'], row2['y']]) <= distance_tolerance*frame_diff:
                if row2['bgd_subtracted_masked_sum'] > 0:
                    results.append(row2)
    
    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    result_df_2 = result_df.drop_duplicates().reset_index(drop=True)
    result_df_2['frame'] = result_df_2.loc[:,'frame'].astype(int)
    
    return result_df_2

# Function to calculate the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to find the longest track
def find_longest_track(df):
    track_lengths = df.groupby('track_id').size()
    longest_track_id = track_lengths.idxmax()
    return df[df['track_id'] == longest_track_id].copy()

# Function to find the next nearest track in the forward direction
def find_next_track_forward(df, end_frame, end_x, end_y,distance_tolerance):
    
    candidate_tracks_bool = (df.groupby('track_id')['frame'].min() > end_frame).reset_index().rename(columns={'frame':'keep'})
    candidate_tracks = df.merge(candidate_tracks_bool,on='track_id')
    candidate_tracks = candidate_tracks[candidate_tracks['keep'] == True]
    candidates = candidate_tracks.drop(columns=['keep'])
    
    distance_l=[]
    for _,row in candidates.iterrows():
        distance = euclidean([end_x, end_y], [row['x'], row['y']])
        distance_l.append(distance)
    candidates.loc[:,'distance'] = distance_l
    candidates.loc[:,'distance_time_score'] = candidates['distance'] + abs(candidates['frame']-end_frame)*distance_tolerance
    
    if not candidates.empty:
        next_track_id = candidates.groupby('track_id')['distance_time_score'].min().idxmin()
        return df[df['track_id'] == next_track_id]
    
    return pd.DataFrame()

# Function to find the next nearest track in the backward direction
def find_next_track_backward(df, start_frame, start_x, start_y,distance_tolerance):
    
    candidate_tracks_bool = (df.groupby('track_id')['frame'].max() < start_frame).reset_index().rename(columns={'frame':'keep'})
    candidate_tracks = df.merge(candidate_tracks_bool,on='track_id')
    candidate_tracks = candidate_tracks[candidate_tracks['keep'] == True]
    candidates = candidate_tracks.drop(columns=['keep'])
    
    distance_l=[]
    for _,row in candidates.iterrows():
        distance = euclidean([start_x, start_y], [row['x'], row['y']])
        distance_l.append(distance)
    candidates.loc[:,'distance'] = distance_l
    candidates.loc[:,'distance_time_score'] = candidates['distance'] + abs(candidates['frame']-start_frame)*distance_tolerance
        
    if not candidates.empty:
        next_track_id = candidates.groupby('track_id')['distance_time_score'].min().idxmin()
        return df[df['track_id'] == next_track_id]
    
    return pd.DataFrame()

# Main function to join tracks
def join_tracks(df,distance_tolerance):
    joined_tracks = pd.DataFrame()
    
    # Find the longest track
    current_track = find_longest_track(df)
    joined_tracks = pd.concat([joined_tracks, current_track])
    df = df[df['track_id'] != current_track['track_id'].iloc[0]]
    
    # Forward direction
    while not current_track.empty:
        end_frame = current_track['frame'].max()
        end_spot = current_track[current_track['frame'] == end_frame].iloc[0]
        next_track = find_next_track_forward(df, end_frame, end_spot['x'], end_spot['y'],distance_tolerance)
        if next_track.empty:
            break
        joined_tracks = pd.concat([joined_tracks, next_track])
        df = df[df['track_id'] != next_track['track_id'].iloc[0]]
        current_track = next_track
    
    # Backward direction
    current_track = find_longest_track(joined_tracks)
    while not current_track.empty:
        start_frame = current_track['frame'].min()
        start_spot = current_track[current_track['frame'] == start_frame].iloc[0]
        next_track = find_next_track_backward(df, start_frame, start_spot['x'], start_spot['y'],distance_tolerance)
        if next_track.empty:
            break
        joined_tracks = pd.concat([joined_tracks, next_track])
        df = df[df['track_id'] != next_track['track_id'].iloc[0]]
        current_track = next_track
    
    return joined_tracks

# Function to repeat join_tracks until no more tracks can be joined
def repeat_join_tracks(df,distance_tolerance = 4):
    result_l = []
    while len(df['track_id'].unique()) > 1:
        joined_tracks = join_tracks(df,distance_tolerance)
        joined_tracks['state'] = 1
        result_l.append(joined_tracks.sort_values(by='frame'))
        df = df[~df['track_id'].isin(joined_tracks['track_id'])]
    
    # Only add potential bursting alleles that burst at least twice
    return result_l

def keep_long_trks(df):
    # If tracks are overlapping, keep the longest one so that there is only one track ongoing at a given time
    # Step 1: Find the start and end frames for each track
    track_ranges = df.groupby('track_id')['frame'].agg(['min', 'max']).reset_index()
    track_ranges.columns = ['track_id', 'start_frame', 'end_frame']
    
    # Step 2: Calculate length and sort tracks by length (descending)
    track_ranges['length'] = track_ranges['end_frame'] - track_ranges['start_frame']
    track_ranges = track_ranges.sort_values(by='length', ascending=False)
    
    # Step 3: Remove overlapping tracks, keeping the longest    
    non_overlapping_tracks_l = []
    
    for _, row in track_ranges.iterrows():
        start = row['start_frame']
        end = row['end_frame']
        overlap = False
        
        # Check for overlap with any already selected track
        for selected in non_overlapping_tracks_l:
            n_start = selected['start_frame']
            n_end = selected['end_frame']
            if not (end < n_start or start > n_end):
                overlap = True
                break
        
        if not overlap:
            non_overlapping_tracks_l.append(row)
    
    non_overlapping_tracks = pd.DataFrame(non_overlapping_tracks_l)

    # Step 4: Filter the original dataframe
    non_overlapping_track_ids = non_overlapping_tracks['track_id'].tolist()
    df_2 = df[df['track_id'].isin(non_overlapping_track_ids)].copy()
    df_2['state'] = 1
    
    return df_2

def fill_missing_frames(track_df, frame_l,valid_mask_l):
    '''

    Parameters
    ----------
    df : pandas dataframe
        contains tracks with length of at least two.
    movie_length : int
        length of the entire tif movie being considered.

    Returns
    -------
    df : pandas dataframe
        filled out dataframe that now puts spot at interpolated positions 
        when burst is OFF.

    '''
    # Step 1: Create a DataFrame with all frames from 0 to movie_length - 1
    start = frame_l[0]
    end = frame_l[-1]
    all_frames = pd.DataFrame({'frame': np.arange(start,end+1)})
    
    # Merge the original DataFrame with the all_frames DataFrame
    df = pd.merge(all_frames, track_df, on='frame', how='left')
    
    # Step 2: Fill missing frames before the first found spot
    first_spot_frame = df.dropna().iloc[0]['frame'].astype(int) - start
    
    # Only do if the first spot is not in the first frame
    if first_spot_frame != 0:
        for frame in np.arange(first_spot_frame):
        
            # Step 5: Get the valid positions
            valid_positions = np.argwhere(valid_mask_l[frame])
            
            if len(valid_positions) == 0:
                raise ValueError("No valid positions found. The object might be too small for the given radius.")
            
            # Step 6: Randomly select a valid position
            y, x = valid_positions[np.random.choice(valid_positions.shape[0])]
            df.loc[frame,['y','x']] = y, x
    else:
        pass
        
    # Step 3: Fill missing frames after the last found spot
    last_spot_frame = df.dropna().iloc[-1]['frame'].astype(int)- start
    
    # Only do if the last spot is not in the last frame
    if last_spot_frame != end:
        for frame in np.arange(last_spot_frame+1,len(df)):
            
            # Step 5: Get the valid positions
            valid_positions = np.argwhere(valid_mask_l[frame])
            
            if len(valid_positions) == 0:
                raise ValueError("No valid positions found. The object might be too small for the given radius.")
            
            # Step 6: Randomly select a valid position
            y, x = valid_positions[np.random.choice(valid_positions.shape[0])]
            df.loc[frame,['y','x']] = y, x
    else:
        pass
        
    # Step 4: Linearly interpolate coordinates for missing frames between found spots
    df['y'] = df['y'].interpolate().round().astype(int)
    df['x'] = df['x'].interpolate().round().astype(int)
    
    return df

def fill_trk_ids_states(df):
    # Forward fill to fill NaNs with the preceding non-NaN values
    forward_fill = df['track_id'].ffill()
    
    # Backward fill to fill NaNs with the succeeding non-NaN values
    backward_fill = df['track_id'].bfill()
    
    # Fill only the NaNs that are between non-NaN values
    df['filled_track_id'] = df['track_id']
    df['filled_track_id'] = np.where(df['track_id'].isna(), 
                                   np.where(forward_fill == backward_fill, forward_fill, np.nan), 
                                   df['track_id'])
    df['filled_state'] = np.where(df['state'].isna(), 
                                      np.where(forward_fill == backward_fill, 1, np.nan), 
                                      df['state'])
    
    df['filled_state'] = df['filled_state'].fillna(value=0)
    
    return df

def recalc_spot_sums(df,msked_nuc_l,sigma_2):
    
    y=df['y']
    x=df['x']
    
    summed_value_l=[]
    
    for msk_img, y, x in zip(msked_nuc_l,y,x):
        
        summed_value = get_masked_sum(msk_img, y, x, sigma_2)
        summed_value_l.append(summed_value)
    
    df['masked_sum'] = summed_value_l
    df['bgd_subtracted_masked_sum'] = df['masked_sum'] - df['mean_bgd_sum']
    
    return df

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
        
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
        
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def erode_tracks(df, threshold_adj):
    
    df = df.reset_index(drop=True)
    
    # Calculate the threshold based on the standard deviation of the intensity values
    std_dev = df['bgd_subtracted_rand_masked_std']
    df['erode_threshold'] = threshold_adj * std_dev
    
    # Find all unique track_ids
    track_ids = df['filled_track_id'].dropna().unique()
    
    for track_id in track_ids:
        
        # Find the first and last index of each active track
        track_indices = df[df['filled_track_id'] == track_id].index
        first_idx = track_indices.min()
        last_idx = track_indices.max()
        
        # Look backward in time from the last index of the current track
        curr_last_idx = last_idx
        while curr_last_idx >= first_idx and df.loc[curr_last_idx, 'smoothened_bgd_subtracted_masked_sum'] <= df.loc[curr_last_idx,'erode_threshold']:
            df.loc[curr_last_idx, 'filled_track_id'] = np.nan
            df.loc[curr_last_idx, 'filled_state'] = 0
            curr_last_idx -= 1
            
        # Look forward in time from the first index of the current track
        curr_first_idx = first_idx
        while curr_first_idx < curr_last_idx and df.loc[curr_first_idx, 'smoothened_bgd_subtracted_masked_sum'] <= df.loc[curr_first_idx,'erode_threshold']:
            df.loc[curr_first_idx, 'filled_track_id'] = np.nan
            df.loc[curr_first_idx, 'filled_state'] = 0
            curr_first_idx += 1

    return df

def dilate_tracks(df, threshold_adj):
    
    df = df.reset_index(drop=True)
    
    # Calculate the threshold based on the standard deviation of the intensity values
    std_dev = df['bgd_subtracted_rand_masked_std']
    df['dilate_threshold'] = threshold_adj * std_dev
    
    # Find all unique track_ids
    track_ids = df['filled_track_id'].dropna().unique()
    
    for count,track_id in enumerate(np.sort(track_ids)):
        
        # Find the first and last index of each active track
        track_indices = df[df['filled_track_id'] == track_id].index
        first_idx = track_indices.min()
        last_idx = track_indices.max()
        
        if first_idx == df.index.min() or last_idx == df.index.max():
            continue
        else:
            pass
        
        if count == 0:
            
            # Look forward in time from the last index of the current track
            next_track_indices = df[df['filled_track_id'] == np.sort(track_ids)[count+1]].index
            next_track_first_idx = next_track_indices.min()

            j = last_idx + 1
            
            while j < next_track_first_idx and df.loc[j, 'smoothened_bgd_subtracted_masked_sum'] >= df.loc[j,'dilate_threshold']:
                df.loc[j, 'filled_track_id'] = track_id
                df.loc[j, 'filled_state'] = 1
                j += 1

        elif count > 0 and count != len(track_ids) -1:
            
            # Look forward in time
            next_track_indices = df[df['filled_track_id'] == np.sort(track_ids)[count+1]].index
            next_track_first_idx = next_track_indices.min()

            j = last_idx + 1
            
            while j < next_track_first_idx and df.loc[j, 'smoothened_bgd_subtracted_masked_sum'] >= df.loc[j,'dilate_threshold']:
                df.loc[j, 'filled_track_id'] = track_id
                df.loc[j, 'filled_state'] = 1
                j += 1
            
            # Look backward in time
            prev_track_indices = df[df['filled_track_id'] == np.sort(track_ids)[count-1]].index
            prev_track_last_idx = prev_track_indices.max()

            k = first_idx - 1
            
            while k > prev_track_last_idx and df.loc[k, 'smoothened_bgd_subtracted_masked_sum'] >= df.loc[k,'dilate_threshold']:
                df.loc[k, 'filled_track_id'] = track_id
                df.loc[k, 'filled_state'] = 1
                k -= 1

        elif count == len(track_ids) -1:
            
            # Look backward in time
            prev_track_indices = df[df['filled_track_id'] == np.sort(track_ids)[count-1]].index
            prev_track_last_idx = prev_track_indices.max()

            k = first_idx - 1
            
            while k > prev_track_last_idx and df.loc[k, 'smoothened_bgd_subtracted_masked_sum'] >= df.loc[k,'dilate_threshold']:
                df.loc[k, 'filled_track_id'] = track_id
                df.loc[k, 'filled_state'] = 1
                k -= 1
        
        else:
            pass

    return df

def merge_close_tracks(df,closed_gap_length=1):
    
    df = df.reset_index(drop=True)

    # Find all unique track_ids
    track_ids = df['filled_track_id'].dropna().unique()
    
    # Get start and ends of all tracks
    start_l=[]
    end_l=[]
    
    for trk_id in track_ids:
        trk_start = df[df['filled_track_id'] == trk_id].index.min()
        trk_end = df[df['filled_track_id'] == trk_id].index.max()
        
        start_l.append(trk_start)
        end_l.append(trk_end)
    
    # Get the gaps between the tracks
    gap_l=[]
    
    for i in np.arange(len(start_l)-1):
        
        next_track_first_idx = start_l[i+1]        
        curr_track_last_idx = end_l[i]
        
        gap = next_track_first_idx - curr_track_last_idx - 1
        
        gap_l.append(gap)
    
    # Get the start and end indices of gaps that fall below the threshold gap
    gap_start_end = []
    start = None

    for i, num in enumerate(gap_l):
        if num <= closed_gap_length:
            if start is None:
                start = i
        else:
            if start is not None:
                gap_start_end.append((start, i - 1))
                start = None

    # If the last number in the list was below the threshold, close the range
    if start is not None:
        gap_start_end.append((start, len(gap_l) - 1))
        
    for ranges in gap_start_end:
        gap_start = ranges[0]
        gap_end = ranges[1]
        
        # Get the track id that should be propagated
        combined_track = track_ids[gap_start]
        new_start = start_l[gap_start]
        new_end = end_l[gap_end+1]
        
        # Set the new track id and merge
        df.loc[new_start:new_end,'filled_track_id'] = combined_track
        df.loc[new_start:new_end,'filled_state'] = 1

    
    return df

def replace_short_on(df,threshold=3):
    
    # Group by the 'group_column' and get the group sizes
    group_counts = df.groupby('filled_track_id').size()
    
    # Identify groups below the threshold
    groups_below_threshold = group_counts[group_counts < threshold].index

    # Apply the conditions to change the column values
    df.loc[df['filled_track_id'].isin(groups_below_threshold), 'filled_state'] = 0
    df.loc[df['filled_track_id'].isin(groups_below_threshold), 'filled_track_id'] = np.nan
    
    return df

#%%

def build_spots_df(frame_l,masked_img_l,masked_img_norm_l,nuc_area_l,max_sigma,sigma_2):
    # For each frame in the movie, segment the nucleus,
    # detect bright spots using LoG method,
    # Get the total intensity in the spots
    # Get the average background spot total intensity
    # And random spot total intensity
    
    spots_l = []
    bgd_l=[]
    valid_mask_l=[]

    bgd_subtracted_rand_masked_sum_l_2= []
    avg_bgd_subtracted_rand_masked_sum_l_2=[]
    spots_n_l =[]
    
    for frame,masked_image,masked_norm_image,nuc_area in zip(frame_l,masked_img_l,masked_img_norm_l,nuc_area_l):
        # Detect the blobs using Laplacian of Gaussian method
        spots = feature.blob_log(masked_norm_image, max_sigma=max_sigma)
        
        if len(spots) > 0:
            df = pd.DataFrame(spots, columns=["y", "x", "sigma"])
        else:
            df = pd.DataFrame(columns=["y", "x", "sigma"])
            df["y"] = np.nan
            df["x"] = np.nan
            df["sigma"] = np.nan
            
        df["frame"] = frame
        df["sigma_2"] = sigma_2 # second sigma for use later area intensity sum measures
        df["nuc_area"] = nuc_area
        
        # Get the intensity measurement inside the spots if there are any found
        existing_centers = [(int(y), int(x)) for y,x in zip(df['y'],df['x'])]
        msk_sum_l=[]
    
        if len(df) > 0:
            for center in existing_centers:
                (y, x) = center
                msk_sum = get_masked_sum(masked_image,y,x,sigma_2)
                msk_sum_l.append(msk_sum)
        
        else:
            df['masked_sum'] = np.nan
        
        # Get the average background spot total intensity
        bgd_df,valid_mask = bgd_msk_sum(masked_image,frame,sigma_2)
        
        valid_mask_l.append(valid_mask)
                
        # Get intensity measurement inside a random spot inside the nucleus
        rand_positions = get_random_disk_centers(masked_image, existing_centers, sigma_2)
        
        bgd_subtracted_rand_masked_sum_l = []
        
        # From about a third of the valid random positions inside the nucleus outside the zones of existing detected spots
        # Get the background subtracted intensity sum of the random positions
        for rand_position in rand_positions:
            rand_y = rand_position[0]
            rand_x = rand_position[1]
        
            rand_masked_sum = get_masked_sum(masked_image, rand_y, rand_x, sigma_2)
            bgd_subtracted_rand_masked_sum = rand_masked_sum - bgd_df['mean_bgd_sum'][0] # This is just a constant
            bgd_subtracted_rand_masked_sum_l.append(bgd_subtracted_rand_masked_sum)
            bgd_subtracted_rand_masked_sum_l_2.append(bgd_subtracted_rand_masked_sum)
            
        # Get the background subtracted sum of one random spot in this frame
        bgd_df['frame'] = frame
        bgd_df['rand_y'] = rand_y
        bgd_df['rand_x'] = rand_x
        
        bgd_df['rand_masked_sum'] = get_masked_sum(masked_image,rand_y,rand_x,sigma_2)
        bgd_df['bgd_subtracted_rand_masked_sum'] = bgd_df['rand_masked_sum'] - bgd_df['mean_bgd_sum']

        # This is to use later for identifying a real burst
        bgd_df['bgd_subtracted_rand_masked_std'] = np.std(bgd_subtracted_rand_masked_sum_l)        
        mean_bgd_sum = bgd_df['mean_bgd_sum'][0]
        msk_sum_l_2 = [msk_sum - mean_bgd_sum for msk_sum in msk_sum_l]
        
        df['masked_sum'] = msk_sum_l
        df['bgd_subtracted_masked_sum'] = msk_sum_l_2
        
        spots_l.append(df)
        bgd_l.append(bgd_df)
        
        spots_n_l.append(len(df))
        avg_bgd_subtracted_rand_masked_sum_l_2.append(np.mean(bgd_subtracted_rand_masked_sum_l))
    
    fig,ax=plt.subplots()
    plt.hist(x=avg_bgd_subtracted_rand_masked_sum_l_2)
    plt.show()
    
    fig,ax=plt.subplots()
    plt.scatter(x=spots_n_l,y=avg_bgd_subtracted_rand_masked_sum_l_2)
    plt.xlabel('# of potential spots')
    plt.ylabel('Mean of random spot total intensity')
    plt.ylim([0,150])

    plt.show()
        
    filtered_spots_l = [spot_df for spot_df in spots_l if not spot_df.empty and not spot_df.isna().all().all()]

    # Concatenate the filtered dataframes
    if filtered_spots_l:
        spots_df = pd.concat(filtered_spots_l)
    else:
        spots_df = pd.DataFrame()  # Empty dataframe if no valid dataframes found
        
    spots_df=spots_df.reset_index(drop=True)
    
    spots_bgd_df = pd.concat(bgd_l)
    spots_bgd_df=spots_bgd_df.reset_index(drop=True)
        
    return spots_df,spots_bgd_df,valid_mask_l
