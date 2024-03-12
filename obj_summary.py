#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:29:23 2022

@author: phillipyeh
"""
import os
#import shapely
#from shapely.ops import orient
import itertools
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
#from metpy.units import units
from datetime import datetime #, timedelta, timezone
import scipy.ndimage as ndi
#from skimage.draw import ellipse
from skimage.measure import label, regionprops_table#, EllipseModel
from skimage.morphology import disk
import pyart

def Z_to_S(refl):
    '''
    Conversion of equivalent reflectivity (Ze) to snow rate (S)
    Equation taken from Rasmussen and Dixon (2003)

    Parameters
    ----------
    Ze : array-like
        The input field, in dBZ.

    Returns
    -------
    S : array-like
        The estimated snow rate, in mm h-1

    '''
    Ze = 10**(refl/10.) # first convert from dBZ to Z in mm6 m-3
    S = (Ze / 57.3)**(1/1.67)
    return S

def rolling_window_arr(arr_in, thresh, box_dim, rollstep, min_thresh=0, max_thresh=None, do_smoothing=False):
    '''
    Calculates the desired quantile in a 2-D rolling (moving) window with
    stepsize > 1

    Parameters
    ----------
    arr_in : array-like
        Input array. A 2-D array (M,N) is expected.
    thresh : float
        Quantile threshold to use.
    box_dim : int
        Size of window box. A square box is assumed.
    rollstep : int
        Number of steps to shift the window forward, assumed to be the same
        in each dimension.
    min_thresh : float
        Minimum reflectivity required for the threshold; lower threshold values
        are either masked or replaced with this value. Default is 0.

    Returns
    -------
    z_out : array-like
        The resulting array of threshold values, with the same dimensions as
        the input array arr_in.

    '''
    if rollstep > box_dim:
        raise ValueError('Invalid slice: {} is greater than {}.'.format(rollstep, box_dim))
    # first get starting point so function is symmetric
    n_iters = n_iters=[(dims-box_dim)//rollstep for dims in arr_in.shape]
    s_stag = [(dims - (rollstep*n + box_dim)) // 2 for dims,n in zip(arr_in.shape,n_iters)]
    z_thresh = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in, (box_dim,box_dim))[s_stag[0]::rollstep,s_stag[1]::rollstep], thresh, axis=(-2,-1))
    # array shapes
    z_dim0, z_dim1 = z_thresh.shape
    # define new array that will be filled in with values
    z_out = np.zeros_like(arr_in) + min_thresh
    # get indices that will be used to fill in values
    ll_start = [s + ((box_dim - rollstep) // 2) for s in s_stag]
    ij_main = [l + np.arange(rollstep) for l in ll_start]
    # loop through indices to fill values
    loop_main = itertools.product(ij_main[0],ij_main[1])
    for i,j in loop_main:
        z_out[i:i+rollstep*z_dim0:rollstep,j:j+rollstep*z_dim1:rollstep]=z_thresh
    # fill in remaining values on the outside
    max_0 = np.arange(ij_main[0][-1],ij_main[0][-1]+rollstep*z_dim0,rollstep)[-1]
    max_1 = np.arange(ij_main[1][-1],ij_main[1][-1]+rollstep*z_dim1,rollstep)[-1]
    z_top = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:box_dim,:], (box_dim,box_dim))[:,::rollstep], thresh, axis=(-2,-1))
    z_left = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,:box_dim], (box_dim,box_dim))[::rollstep], thresh, axis=(-2,-1))
    z_bottom = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[-box_dim:,:], (box_dim,box_dim))[:,::rollstep], thresh, axis=(-2,-1))
    z_right = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,-box_dim:], (box_dim,box_dim))[::rollstep], thresh, axis=(-2,-1))
    # box_dim and rollstep are same in x-y direction
    # use 2 loops for i and j
    for i in ij_main[0]:
        z_out[i:i+rollstep*z_dim0:rollstep,:ll_start[1]]=z_left
        z_out[i:i+rollstep*z_dim0:rollstep,max_1:]=z_right
    for j in ij_main[1]:
        z_out[:ll_start[0],j:j+rollstep*z_dim1:rollstep]=z_top
        z_out[max_0:,j:j+rollstep*z_dim1:rollstep]=z_bottom

    # finally, handle the corners
    z_UL = np.nanquantile(arr_in[:box_dim,:box_dim], thresh)
    z_LL = np.nanquantile(arr_in[:box_dim,-box_dim:], thresh)
    z_UR = np.nanquantile(arr_in[-box_dim:,:box_dim], thresh)
    z_LR = np.nanquantile(arr_in[-box_dim:,-box_dim:], thresh)
    z_out[:ll_start[0],:ll_start[1]] = z_UL
    z_out[:ll_start[0],max_1:] = z_LL
    z_out[max_0:,:ll_start[1]] = z_UR
    z_out[max_0:,max_1:] = z_LR
    if max_thresh is not None:
        z_out = np.where(z_out<max_thresh,z_out,0.5*(max_thresh+z_out))
    if do_smoothing:
        z_out = mpcalc.smooth_circular(z_out, rollstep//2, 2)
    z_out[np.isnan(arr_in)] = float('nan')
    return z_out

def rolling_window_var(arr_in, thresh, box_dim, rollstep, var_thresh=45, min_thresh=0, max_thresh=None):
    if rollstep > box_dim:
        raise ValueError('Invalid slice: {} is greater than {}.'.format(rollstep, box_dim))
    import itertools
    # first get starting point so function is symmetric
    n_iters = n_iters=[(dims-box_dim)//rollstep for dims in arr_in.shape]
    s_stag = [(dims - (rollstep*n + box_dim)) // 2 for dims,n in zip(arr_in.shape,n_iters)]
    # the thresholds
    z_thresh = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in, (box_dim,box_dim))[s_stag[0]::rollstep,s_stag[1]::rollstep], thresh, axis=(-2,-1))
    z_alt = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in, (box_dim,box_dim))[s_stag[0]::rollstep,s_stag[1]::rollstep], 0.750, axis=(-2,-1))
    z_var = np.nanvar(np.lib.stride_tricks.sliding_window_view(
        arr_in, (box_dim,box_dim))[s_stag[0]::rollstep,s_stag[1]::rollstep], ddof=1, axis=(-2,-1))
    # array shapes
    z_dim0, z_dim1 = z_thresh.shape
    # define new array that will be filled in with values
    z_out = np.zeros_like(arr_in) + min_thresh
    z_2 = np.zeros_like(arr_in) + min_thresh
    var_out = np.zeros_like(arr_in)
    # get indices that will be used to fill in values
    ll_start = [s + ((box_dim - rollstep) // 2) for s in s_stag]
    ij_main = [l + np.arange(rollstep) for l in ll_start]
    loop_main = itertools.product(ij_main[0],ij_main[1])
    for i,j in loop_main:
        z_out[i:i+rollstep*z_dim0:rollstep,j:j+rollstep*z_dim1:rollstep]=z_thresh
        z_2[i:i+rollstep*z_dim0:rollstep,j:j+rollstep*z_dim1:rollstep]=z_alt
        var_out[i:i+rollstep*z_dim0:rollstep,j:j+rollstep*z_dim1:rollstep]=z_var

    # fill in remaining zero values on the outside
    max_0 = np.arange(ij_main[0][-1],ij_main[0][-1]+rollstep*z_dim0,rollstep)[-1]
    max_1 = np.arange(ij_main[1][-1],ij_main[1][-1]+rollstep*z_dim1,rollstep)[-1]
    # next, handle bottom/right edges
    z_top = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:box_dim,:], (box_dim,box_dim))[:,::rollstep], thresh, axis=(-2,-1))
    z_left = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,:box_dim], (box_dim,box_dim))[::rollstep], thresh, axis=(-2,-1))
    z_bottom = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[-box_dim:,:], (box_dim,box_dim))[:,::rollstep], thresh, axis=(-2,-1))
    z_right = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,-box_dim:], (box_dim,box_dim))[::rollstep], thresh, axis=(-2,-1))
    z_tv = np.nanvar(np.lib.stride_tricks.sliding_window_view(
        arr_in[:box_dim,:], (box_dim,box_dim))[:,::rollstep], ddof=1, axis=(-2,-1))
    z_lv = np.nanvar(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,:box_dim], (box_dim,box_dim))[::rollstep], ddof=1, axis=(-2,-1))
    z_bv = np.nanvar(np.lib.stride_tricks.sliding_window_view(
        arr_in[-box_dim:,:], (box_dim,box_dim))[:,::rollstep], ddof=1, axis=(-2,-1))
    z_rv = np.nanvar(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,-box_dim:], (box_dim,box_dim))[::rollstep], ddof=1, axis=(-2,-1))
    z_t2 = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:box_dim,:], (box_dim,box_dim))[:,::rollstep], 0.750, axis=(-2,-1))
    z_l2 = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,:box_dim], (box_dim,box_dim))[::rollstep], 0.750, axis=(-2,-1))
    z_b2 = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[-box_dim:,:], (box_dim,box_dim))[:,::rollstep], 0.750, axis=(-2,-1))
    z_r2 = np.nanquantile(np.lib.stride_tricks.sliding_window_view(
        arr_in[:,-box_dim:], (box_dim,box_dim))[::rollstep], 0.750, axis=(-2,-1))
    # since box_dim and rollstep are same in x-y direction, can use same loop
    for i in ij_main[0]:
        z_out[i:i+rollstep*z_dim0:rollstep,:ll_start[1]]=z_left
        z_out[i:i+rollstep*z_dim0:rollstep,max_1:]=z_right
        var_out[i:i+rollstep*z_dim0:rollstep,:ll_start[1]]=z_lv
        var_out[i:i+rollstep*z_dim0:rollstep,max_1:]=z_rv
        z_2[i:i+rollstep*z_dim0:rollstep,:ll_start[1]]=z_l2
        z_2[i:i+rollstep*z_dim0:rollstep,max_1:]=z_r2
    for i in ij_main[1]:
        z_out[:ll_start[0],i:i+rollstep*z_dim1:rollstep]=z_top
        z_out[max_0:,i:i+rollstep*z_dim1:rollstep]=z_bottom
        var_out[:ll_start[0],i:i+rollstep*z_dim1:rollstep]=z_tv
        var_out[max_0:,i:i+rollstep*z_dim1:rollstep]=z_bv
        z_2[:ll_start[0],i:i+rollstep*z_dim1:rollstep]=z_t2
        z_2[max_0:,i:i+rollstep*z_dim1:rollstep]=z_b2
    # finally, handle the corners
    z_UL = np.nanquantile(arr_in[:box_dim,:box_dim], thresh)
    z_LL = np.nanquantile(arr_in[:box_dim,-box_dim:], thresh)
    z_UR = np.nanquantile(arr_in[-box_dim:,:box_dim], thresh)
    z_LR = np.nanquantile(arr_in[-box_dim:,-box_dim:], thresh)
    v_UL = np.nanvar(arr_in[:box_dim,:box_dim], ddof=1)
    v_LL = np.nanvar(arr_in[:box_dim,-box_dim:], ddof=1)
    v_UR = np.nanvar(arr_in[-box_dim:,:box_dim], ddof=1)
    v_LR = np.nanvar(arr_in[-box_dim:,-box_dim:], ddof=1)
    n_UL = np.nanquantile(arr_in[:box_dim,:box_dim], 0.750)
    n_LL = np.nanquantile(arr_in[:box_dim,-box_dim:], 0.750)
    n_UR = np.nanquantile(arr_in[-box_dim:,:box_dim], 0.750)
    n_LR = np.nanquantile(arr_in[-box_dim:,-box_dim:], 0.750)
    z_out[:ll_start[0],:ll_start[1]] = z_UL
    z_out[:ll_start[0],max_1:] = z_LL
    z_out[max_0:,:ll_start[1]] = z_UR
    z_out[max_0:,max_1:] = z_LR
    var_out[:ll_start[0],:ll_start[1]] = v_UL
    var_out[:ll_start[0],max_1:] = v_LL
    var_out[max_0:,:ll_start[1]] = v_UR
    var_out[max_0:,max_1:] = v_LR
    z_2[:ll_start[0],:ll_start[1]] = n_UL
    z_2[:ll_start[0],max_1:] = n_LL
    z_2[max_0:,:ll_start[1]] = n_UR
    z_2[max_0:,max_1:] = n_LR
    # replace values based on variance
    if var_thresh is None:
        # set the variance threshold to be the upper half of variance
        var_thresh = np.nanmedian(var_out)
        print(var_thresh)
        z_out = np.where(var_out<var_thresh,z_out,z_2)
    else:
        z_out = np.where(var_out<var_thresh,z_out,z_2)
    if max_thresh is not None:
        z_out = np.where(z_out<max_thresh,z_out,0.5*(max_thresh+z_out))
    z_smoothed = mpcalc.smooth_circular(z_out, rollstep//2, 2)
    #z_out = np.nan_to_num(z_out, nan=min_thresh)
    #return np.where(z_out>min_thresh,z_out,min_thresh)
    z_out[np.isnan(arr_in)] = float('nan')
    z_smoothed[np.isnan(arr_in)] = float('nan')
    return z_out, z_smoothed

def isolate_by_thresh(arr_in, min_refl=0., dev_thresh=1.25):
    '''
    Reflectivity isolation algorithm used in Radford et al. (2019)

    arr_in : input array
    dev_thresh : standard deviation above which to calculate threshold
    min_refl : reflectivity threshold to use as a first pass
    '''
    # convert input array into numpy
    arr_in = np.asanyarray(arr_in)
    
    # initialize output array
    out = np.zeros_like(arr_in)

    # binary array where array exceeds first pass threshold
    pass_1 = arr_in >= min_refl
    label_1 = label(pass_1, connectivity=pass_1.ndim)
    # calculate the mean and standard deviation for each labeled region
    for i in range(1, np.amax(label_1)+1): 
        sd = np.nanstd(arr_in[np.nonzero(label_1==i)], ddof=1)
        mn = np.nanmean(arr_in[np.nonzero(label_1==i)])
        out[np.nonzero(label_1==i)] = mn + dev_thresh*sd
    # set values without data to NaN
    out[np.isnan(arr_in)] = float('nan')
    return out

def TRACE3D_RC(intensity_img, thresh, maxdiff=10.):
    '''
    First calculates regions of intense precipitation (ROIPs),
    then isolates reflectivity cores (RCs) for each ROIP.
    TRACE3D Algorithm used in Handwerker (2002)

    Parameters
    ----------
    intensity_img : array-lie
        array with intenstity values, MxN.
    thresh : float
        reflectivity threshold value to use as a first pass.
    maxdiff : TYPE, optional
        reflectivity difference from maximum to use as a new
        threshold in each region. The default is 10.

    Returns
    -------
    arr_out : array-like
        output binary array containing isolated objects.

    '''
    from skimage.measure import label
    # convert input array into numpy
    intensity_img = np.asanyarray(intensity_img)
    binary_arr = intensity_img>=thresh
    label_img = label(binary_arr, connectivity=1)
    # new empty array
    arr_out = np.zeros_like(label_img)
    for l in range(1, np.nanmax(label_img)+1):
        lb = np.where(label_img==l, intensity_img, float('nan'))
        thresh_local = np.nanmax(lb) - maxdiff
        arr_out[lb>=thresh_local] = 1
    return arr_out

def tobac_refl(intensity_img, thresh_list):
    from skimage.measure import label
    # convert input array into numpy
    intensity_img = np.asanyarray(intensity_img)
    bins = [intensity_img>=thresh for thresh in thresh_list]
    label_img = label(bins[0], connectivity=1)
    for bin_arr in bins[1:]:
        label_img = label(watershed_by_arr(label_img, bin_arr, intensity_img))
    return label_img

def watershed_by_arr(thresh_arr, bin2, intensity_img=None, min_area=1):
    '''
    Selects "large" objects with large gaps and performs watershed segmentation
    to reduce the size of those objects. Uses a second binary array to determine
    the basin centers.

    Parameters
    ----------
    thresh_arr : array-like
        labeled array, MxN.
    bin2 : array-like
        second binary array, MxN.
    intensity_img : array-like, optional
        Array with intenstity values, MxN. If provided, the local maxima will be calculated from the
        intensity array rather than from the exact Euclidian distance transform. The default is None.
    min_area : float, optional
        Structuring element used for image erosion. Non-zero elements are considered True. If no
        structuring element is provided, an element is generated with a square connectivity equal
        to one. The default is 1.

    Returns
    -------
    out : array-like
        New labeled array, after performing watershed segmentation
        on large/non-solid objects.

    '''
    from skimage.measure import label
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    # convert input array into numpy
    thresh_arr = np.asanyarray(thresh_arr) # main binary array
    binary_2 = np.asanyarray(bin2) # second array used to find centers, should already be binary array
    # new empty array
    arr_new = np.zeros_like(thresh_arr)
    # get coordinates of labeled objects
    props = regionprops_table(label(binary_2, connectivity=1),
                                properties=('coords',
                                            'area_filled',
                                            'solidity'))

    df = pd.DataFrame(props)
    df_sub = df[df['area_filled']>=min_area]
    for lb in df_sub['coords'].to_list():
        # split indices since we want individual points, not an array
        rows = lb[:,0] # get row indices
        columns = lb[:,1] # get column indices
        arr_new[(rows,columns)] = 1 # include the objects with many gaps, ignoring small objects
    binary_labels = label(arr_new, connectivity=1)
    if intensity_img is not None:
        coords = peak_local_max(intensity_img, threshold_abs=0., labels=binary_labels, num_peaks_per_label=1)
        mask = np.zeros(intensity_img.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        labels = watershed(-intensity_img, markers, mask=thresh_arr>0)
    else:
        distance = ndi.distance_transform_edt(binary_labels)
        coords = peak_local_max(distance, labels=binary_labels, num_peaks_per_label=1)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        labels = watershed(-distance, markers, mask=thresh_arr>0)
    # get union of all arrays
    out = np.where(labels>0, 1000+labels, thresh_arr)
    return out

def get_object_df(label_img, intensity_img, lats, lons, y, x):
    '''
    Used for getting the object dataframe, calculated from the NCstate dataset
    '''
    props = regionprops_table(label_img, intensity_image=intensity_img,
                                properties=('orientation',
                                            'major_axis_length',
                                            'minor_axis_length',
                                            'area_filled',
                                            'eccentricity',
                                            'intensity_max',
                                            'intensity_mean',
                                            'intensity_min',
                                            'coords'))
    df = pd.DataFrame(props)
    coords_arr = df.pop('coords')

    geod = pyproj.Geod(ellps="WGS84")
    lon_list = []
    lat_list = []
    xc_list = []
    yc_list = []
    maj_list = []
    min_list = []
    area_list = []
    for i, (coords, bool_err) in enumerate(zip(coords_arr, np.isnan(df['intensity_mean']).to_list())):
        # first handle the lat/lon calculations
        lat_points = lats.values[coords[:,0],coords[:,1]]
        lon_points = lons.values[coords[:,0],coords[:,1]]
        points_latlon = shapely.geometry.MultiPoint([(lon0,lat0) for lon0,lat0 in zip(lon_points,lat_points)])
        lon_list.append(points_latlon.centroid.x)
        lat_list.append(points_latlon.centroid.y)
        points = shapely.geometry.MultiPoint([(x0,y0) for x0,y0 in zip(x.values[coords[:,1]],y.values[coords[:,0]])])
        xc = points.centroid.x
        yc = points.centroid.y
        xc_list.append(xc)
        yc_list.append(yc)
        # next handle NaNs in the intensity values
        if bool_err:
            rows = coords[:,0] # get row indices
            columns = coords[:,1] # get column indices
            arr_tmp = intensity_img[(rows,columns)] # values of region
            df.iloc[i, df.columns.get_loc('intensity_max')] = np.nanmax(arr_tmp)
            df.iloc[i, df.columns.get_loc('intensity_mean')] = np.nanmean(arr_tmp)
            df.iloc[i, df.columns.get_loc('intensity_min')] = np.nanmin(arr_tmp)

    # combine lists into dataframe
    df['cen_x'] = xc_list
    df['cen_y'] = yc_list
    df['cen_lon'] = lon_list
    df['cen_lat'] = lat_list
    df['Length'] = 2*df['major_axis_length']#maj_list
    df['Width'] = 2*df['minor_axis_length']#min_list
    df['Area'] = 4*df['area_filled']#area_list
    df['Aspect'] = df['minor_axis_length'] / df['major_axis_length']
    df['Axis Angle'] = (np.pi/2.) - df['orientation'] # skimage uses y axis as principal axis
    df['Axis Angle'].mask(df['Axis Angle']>np.pi/2., df['Axis Angle']-np.pi, inplace=True) # keep bounds within -90 and 90
    return df
    
def get_obj_idealized(label_img):
    '''
    Used to get the synthetically generated objects dataframe
    Since this is an idealized dataset, there is no geographical x-y information, and the dimensions are in pixel space
    '''
    props = regionprops_table(label_img,
                                properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length',
                                            'area_filled'))
    df = pd.DataFrame(props).sort_values(
        by=['centroid-0','centroid-1'], key=lambda col: np.round(col)).reset_index(drop=True)
    return df

def get_obj_stats(input_vals, binary_arr, method='default', watershed=0, min_area=1, add_arr=None):
    ref_arr, lats, lons, y, x = input_vals
    # structuring element
    kernel = disk(2) #np.ones((3,3))
    # whether image morphology is applied or not
    if method=='default':
        label_img = label(binary_arr, connectivity=1) # default
    elif method=='erosion':
        binary_e30 = ndi.binary_erosion(binary_arr, kernel, iterations=1)
        label_img = label(binary_e30, connectivity=1)
    elif method=='opening':
        binary_open = ndi.binary_opening(binary_arr, kernel)
        label_img = label(binary_open, connectivity=1) # image opening
    elif method=='closing':
        binary_cl = ndi.binary_closing(binary_arr, kernel)
        label_img = label(binary_cl, connectivity=1) # image closing
    else:
        print('invalid method supplied, will use default')
        label_img = label(binary_arr, connectivity=1)
    if watershed==0:
        pass
    elif watershed==1:
        label_img = watershed_by_arr(label_img, add_arr, intensity_img=ref_arr.values, min_area=min_area)
    else:
        raise ValueError('Invalid value given for watershed, valid choices are [0,1]')
    df = get_object_df(label_img, ref_arr.values, lats, lons, y, x)
    return df
