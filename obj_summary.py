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
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mticker
#import matplotlib.dates as mdates
#import matplotlib.patches as mpatches
import scipy.ndimage as ndi
#from skimage.draw import ellipse
from skimage.measure import label, regionprops_table#, EllipseModel
from skimage.morphology import disk
#from skimage.transform import rotate
from radar_cmaps import HomeyerRainbow
import pyart
from pyart.retrieve import conv_strat_yuter

FILEDIR = '/Users/phillipyeh/Documents/python_files/from_NCSU'
SAVEDIR = '/Users/phillipyeh/Documents/python_files/objects_reflectivity'

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

def rolling_window_arr(arr_in, thresh, box_dim, rollstep, min_thresh=0, max_thresh=None):
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
    z_smoothed = mpcalc.smooth_circular(z_out, rollstep//2, 2)
    z_out[np.isnan(arr_in)] = float('nan')
    z_smoothed[np.isnan(arr_in)] = float('nan')
    return z_out, z_smoothed

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

def get_obj_df(label_img):
    props = regionprops_table(label_img,
                                properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length',
                                            'area_filled'))
    df = pd.DataFrame(props).sort_values(
        by=['centroid-0','centroid-1'], key=lambda col: np.round(col)).reset_index(drop=True)
    #.sort_values(by=['area_filled'], ascending=False)
    return df

def get_obj_stats(input_vals, thresh_vals):
    binary_arr = input_vals >= thresh_vals
    label_img = label(binary_arr, connectivity=1)#binary_arr.ndim)
    props = regionprops_table(label_img,
                                properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length',
                                            'area_filled'))
    df = pd.DataFrame(props).sort_values(
        by=['centroid-0','centroid-1'], key=lambda col: np.round(col)).reset_index(drop=True)
    #.sort_values(by=['area_filled'], ascending=False)
    return df

def get_obj_stats2(input_vals, binary_arr):
    label_img = label(binary_arr, connectivity=1)#binary_arr.ndim)
    props = regionprops_table(label_img,
                                properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length',
                                            'area_filled'))
    df = pd.DataFrame(props).sort_values(
        by=['centroid-0','centroid-1'], key=lambda col: np.round(col)).reset_index(drop=True)
    #.sort_values(by=['area_filled'], ascending=False)
    return df

def plot_objects(fid, fields, sfields, rfields, htfields, csfields, tfields, use_smooth=False):
    # unpack variables
    f1, f2, f3, f4, f5 = fields
    s1, s2, s3, s4, s5 = sfields
    r1, r2, r3, r4, r5 = rfields
    h1, h2, h3, h4, h5 = htfields
    c1, c2, c3, c4, c5 = csfields
    t11, t22, t33, t44, t55 = tfields
    if use_smooth:
        t1 = t11[:,1]
        t2 = t22[:,1]
        t3 = t33[:,1]
        t4 = t44[:,1]
        t5 = t55[:,1]
    else:
        t1 = t11[:,0]
        t2 = t22[:,0]
        t3 = t33[:,0]
        t4 = t44[:,0]
        t5 = t55[:,0]

    levs = np.arange(-10,55,2)
    norm = colors.BoundaryNorm(levs, HomeyerRainbow.N)

    fig, axs = plt.subplots(7,5, figsize=(12,16), sharex=True, sharey=True,
                             subplot_kw=dict(box_aspect=1))
    for i in range(7):
        m=axs[i,0].pcolormesh(f1, cmap=HomeyerRainbow, norm=norm, rasterized=True)
        axs[i,1].pcolormesh(f2, cmap=HomeyerRainbow, norm=norm, rasterized=True)
        axs[i,2].pcolormesh(f3, cmap=HomeyerRainbow, norm=norm, rasterized=True)
        axs[i,3].pcolormesh(f4, cmap=HomeyerRainbow, norm=norm, rasterized=True)
        axs[i,4].pcolormesh(f5, cmap=HomeyerRainbow, norm=norm, rasterized=True)
    axs[0,0].set_title('F1: Ellipse Cores')
    axs[0,1].set_title('F2: Multi-Core Ellipse')
    axs[0,2].set_title('F3: Irregular')
    axs[0,3].set_title('F4: Weak Complex')
    axs[0,4].set_title('F5: Strong Complex')
    # Ganetis method contours
    axs[0,0].contour(f1 >= s1, [0.5])
    axs[0,1].contour(f2 >= s2, [0.5])
    axs[0,2].contour(f3 >= s3, [0.5])
    axs[0,3].contour(f4 >= s4, [0.5])
    axs[0,4].contour(f5 >= s5, [0.5])
    # Radford method contours
    axs[1,0].contour(f1 >= r1, [0.5])
    axs[1,1].contour(f2 >= r2, [0.5])
    axs[1,2].contour(f3 >= r3, [0.5])
    axs[1,3].contour(f4 >= r4, [0.5])
    axs[1,4].contour(f5 >= r5, [0.5])
    # tobac method contours
    axs[2,0].contour(h1, [0.5,15,30], colors='k', linewidths=[1,0.7,0.7])
    axs[2,1].contour(h2, [0.5,5,15,25,35], colors='k', linewidths=[1,0.7,0.7,0.7,0.7])
    axs[2,2].contour(h3, [0.5], colors='k')
    axs[2,3].contour(h4, [0.5], colors='k')
    axs[2,4].contour(h5, [0.5], colors='k')
    # convsf method contours
    axs[3,0].contour(c1, [0.5])
    axs[3,1].contour(c2, [0.5])
    axs[3,2].contour(c3, [0.5])
    axs[3,3].contour(c4, [0.5])
    axs[3,4].contour(c5, [0.5])
    # now go through iterations
    for i in range(3):
        axs[i+4,0].contour(f1 >= t1[i], [0.5], linewidths=1.5)
        axs[i+4,1].contour(f2 >= t2[i], [0.5], linewidths=1.5)
        axs[i+4,2].contour(f3 >= t3[i], [0.5], linewidths=1.5)
        axs[i+4,3].contour(f4 >= t4[i], [0.5], linewidths=1.5)
        axs[i+4,4].contour(f5 >= t5[i], [0.5], linewidths=1.5)
    plt.subplots_adjust(left=0.04, right=0.885, bottom=0.04, top=0.928, wspace=0.085, hspace=0.05)
    fig.text(fig.subplotpars.left, 0.97, 'Comparing G18, R19, convsf, 2D-filter, UQ, US-U30', fontsize=15)
    pos1=axs[5,4].get_position()
    ax2 = fig.add_subplot(122, anchor='E')
    ax2.set_position([pos1.x0+0.17, pos1.y0, 0.022, 6*pos1.height],which='both')
    cb=fig.colorbar(m, cax=ax2,
                 orientation='vertical',
                 ticks=np.arange(-10,55,2))
    ax2.tick_params(axis='y', which='major', direction='in', width=1, length=6,
                        labelsize=12, right=True, left=False, pad=2)
    cb.outline.set_linewidth(1.0)
    cb.set_label('Reflectivity     $dBZ$', fontweight='bold', fontsize=13, ha='center', y=0.9)
    if use_smooth:
        fig.savefig(os.path.join(fid,'objects_synthetic_comparison_G18-R19-tobac-convsf_smooth.png'))
    else:
        fig.savefig(os.path.join(fid,'objects_synthetic_comparison_G18-R19-tobac-convsf.png'))
    plt.close()

def plot_by_alg(fignm, fields, sfields, stype=None):
    # unpack variables
    f1, f2, f3, f4, f5 = fields
    s1, s2, s3, s4, s5 = sfields
    fig, axs = plt.subplots(1,5, figsize=(11,3), sharex=True, sharey=True,
                             subplot_kw=dict(box_aspect=1))
    levs = np.arange(-10,55,2)
    norm = colors.BoundaryNorm(levs, HomeyerRainbow.N)
    m=axs[0].pcolormesh(f1, cmap=HomeyerRainbow, norm=norm, rasterized=True)
    axs[1].pcolormesh(f2, cmap=HomeyerRainbow, norm=norm, rasterized=True)
    axs[2].pcolormesh(f3, cmap=HomeyerRainbow, norm=norm, rasterized=True)
    axs[3].pcolormesh(f4, cmap=HomeyerRainbow, norm=norm, rasterized=True)
    axs[4].pcolormesh(f5, cmap=HomeyerRainbow, norm=norm, rasterized=True)
    if stype=='binary':
        # binary field
        axs[0].contour(s1, [0.5])
        axs[1].contour(s2, [0.5])
        axs[2].contour(s3, [0.5])
        axs[3].contour(s4, [0.5])
        axs[4].contour(s5, [0.5])
    elif stype=='tobac':
        # tobac method contours
        axs[0].contour(s1, [0.5,15,30], colors='k', linewidths=[1,0.7,0.7])
        axs[1].contour(s2, [0.5,5,15,25,35], colors='k', linewidths=[1,0.7,0.7,0.7,0.7])
        axs[2].contour(s3, [0.5], colors='k')
        axs[3].contour(s4, [0.5], colors='k')
        axs[4].contour(s5, [0.5], colors='k')
    else:
        # threshold field
        axs[0].contour(f1 >= s1, [0.5])
        axs[1].contour(f2 >= s2, [0.5])
        axs[2].contour(f3 >= s3, [0.5])
        axs[3].contour(f4 >= s4, [0.5])
        axs[4].contour(f5 >= s5, [0.5])

    plt.subplots_adjust(left=0.04, right=0.96, bottom=0.195, top=1.1, wspace=0.11)
    pos1=axs[0].get_position()
    ax2 = fig.add_subplot(212, anchor='N')
    ax2.set_position([pos1.x0, pos1.y0-0.135, 4*pos1.width, 0.055],which='both')
    cb = fig.colorbar(m, cax=ax2,
                        orientation='horizontal',
                        aspect=20,
                        spacing='uniform',
                        ticks=levs[::2],
                        format='%i')
    cb.ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax2.tick_params(axis='x', which='major', direction='in', width=1, length=6,
                    labelsize=12, bottom=True, top=False, pad=2)
    ax2.tick_params(axis='x', which='minor', direction='in', width=0.7, bottom=True, top=False)
    cb.outline.set_linewidth(1.0)
    cb.set_label('Reflectivity    $dBZ$',
                    fontweight='bold',
                    fontsize=13,
                    ha='right',
                    x=1.0)

    plt.savefig(fignm)
    plt.close()

#%% get fields directly from nc files
base2 = xr.open_dataarray(os.path.join(FILEDIR, 'ideal_F1.nc'), engine='netcdf4')
base3 = xr.open_dataarray(os.path.join(FILEDIR, 'ideal_F2.nc'), engine='netcdf4')
base_nc = xr.open_dataarray(os.path.join(FILEDIR, 'ideal_F3.nc'), engine='netcdf4')
base_cl = xr.open_dataarray(os.path.join(FILEDIR, 'ideal_F4.nc'), engine='netcdf4')
base_cv = xr.open_dataarray(os.path.join(FILEDIR, 'ideal_F5.nc'), engine='netcdf4')
#ellipse_sums = 41544.42125107142
#sum_obj = 37052.1485449842
#sum_conv = 53973

KOKX_lat, KOKX_lon, KOKX_alt = pyart.io.nexrad_common.NEXRAD_LOCATIONS['KOKX']
grid_y = {'axis': 'Y', 'data': np.linspace(0, np.shape(base_nc)[0]*1000, np.shape(base_nc)[0]),
          'long_name': 'Y distance on the projection plane from the origin',
          'standard_name': 'projection_y_coordinate', 'units': 'm'}
grid_x = {'axis': 'X', 'data': np.linspace(0, np.shape(base_nc)[1]*1000, np.shape(base_nc)[1]),
          'long_name': 'X distance on the projection plane from the origin',
          'standard_name': 'projection_x_coordinate', 'units': 'm'}
grid_z = {'axis': 'Z', 'data': np.array([10]),
          'long_name': 'Z distance on the projection plane from the origin',
          'positive': 'up', 'standard_name': 'projection_z_coordinate', 'units': 'm'}

cs_list = []
import convsf_fun
for refl in [base2, base3, base_nc, base_cl, base_cv]:
    grid_time = datetime.now()
    grid_meta = dict()
    snrate = Z_to_S(refl)
    snow_dict = {'data': snrate, 'units': 'mm/h', 'long_name': 'estimated snow rate',
                    '_FillValue': -9999.9, 'standard_name': 'snow_rate'}
    save_fields = {'snow': snow_dict}

    # create new grid object to save
    new_grid = pyart.core.Grid(time=grid_time, fields=save_fields, metadata=grid_meta,
                               origin_latitude=KOKX_lat, origin_longitude=KOKX_lon, origin_altitude=KOKX_alt,
                               x=grid_x, y=grid_y, z=grid_z)
    # calculate convsf
    #convsf_heavy_dict = conv_strat_yuter(new_grid, dx=2000, dy=2000, refl_field='snow', dB_averaging=False,
    #                                        always_core_thres=5, bkg_rad_km=40, use_cosine=True, max_diff=1.5,
    #                                        zero_diff_cos_val=5, weak_echo_thres=0, min_dBZ_used=0,
    #                                        val_for_max_conv_rad=10, max_conv_rad_km=2, estimate_offset=1)

    convsf_weak_dict = conv_strat_yuter(new_grid, dx=2000, dy=2000, refl_field='snow', dB_averaging=False,
                                            always_core_thres=5, bkg_rad_km=40, use_cosine=False,
                                            use_addition=False, scalar_diff=1.5, weak_echo_thres=0, min_dBZ_used=0,
                                            val_for_max_conv_rad=10, max_conv_rad_km=2, estimate_offset=1)
    cs_list.append(np.where(convsf_weak_dict['convsf']['data'] > 1.5, 1, 0))

cs_2 = []
for refl in [base2, base3, base_nc, base_cl, base_cv]:
    snrate = Z_to_S(refl)
    convsf_cos_best, convsf_scl_best = convsf_fun._revised_conv_strat(
                                   snrate, 2000, 2000, always_core_thres=5, bkg_rad_km=40,
                                   max_diff=1.5, zero_diff_cos_val=5,
                                   scalar_diff=1.5, use_addition=False, calc_thres=0.75,
                                   weak_echo_thres=0,min_dBZ_used=0, dB_averaging=False,
                                   val_for_max_conv_rad=10, max_conv_rad_km=2)
    convsf_scl_best_filtered = convsf_fun.convsf_filtering(convsf_scl_best, min_size=30)
    cs_2.append(convsf_scl_best_filtered)
#%% calculate areas for Ganetis method
# first get background gradient
bkg_gradient = np.linspace(-5, 25, 601)
# repeat array over other dimension
base_gradient = np.tile(bkg_gradient, (601,1))
# low refl gradient
bkg_low = np.linspace(8, 18, 601)
base_low = np.tile(bkg_low, (601,1)).T
# calculate Ganetis stats
gv1 = np.nanquantile(base2, 0.833, axis=None)
gv2 = np.nanquantile(base3, 0.833, axis=None)
gc1 = np.nanquantile(base_nc, 0.833, axis=None)
gc2 = np.nanquantile(base_cl, 0.833, axis=None)
gc3 = np.nanquantile(base_cv, 0.833, axis=None)
# calculate Radford stats
rv1 = isolate_by_thresh(base2)
rv2 = isolate_by_thresh(base3)
rc1 = isolate_by_thresh(base_nc)
rc2 = isolate_by_thresh(base_cl)
rc3 = isolate_by_thresh(base_cv)
# calculate TRACE3D stats
hv1 = TRACE3D_RC(base2, 18.)
hv2 = TRACE3D_RC(base3, 18.)
hc1 = TRACE3D_RC(base_nc, 18.)
hc2 = TRACE3D_RC(base_cl, 18.)
hc3 = TRACE3D_RC(base_cv, 18.)
# calculate tobac stats
qvals = [0.8, 0.833, 0.875]
bv1 = tobac_refl(base2, [np.nanquantile(base2, q, axis=None) for q in qvals])
bv2 = tobac_refl(base3, [np.nanquantile(base3, q, axis=None) for q in qvals])
bc1 = tobac_refl(base_nc, [np.nanquantile(base_nc, q, axis=None) for q in qvals])
bc2 = tobac_refl(base_cl, [np.nanquantile(base_cl, q, axis=None) for q in qvals])
bc3 = tobac_refl(base_cv, [np.nanquantile(base_cv, q, axis=None) for q in qvals])
'''
# Ganetis
df_e2 = get_obj_stats(base2, gv1)#.loc[lambda df: df['centroid-1'] < 550]
df_e3 = get_obj_stats(base3, gv2)#.loc[lambda df: df['centroid-1'] < 550]
df_r = get_obj_stats(base_nc, gc1)#.loc[lambda df: df['centroid-1'] < 530]
df_l = get_obj_stats(base_cl, gc2).loc[lambda df: df['centroid-0'] < 560]
df_c = get_obj_stats(base_cv, gc3).loc[lambda df: df['centroid-0'] < 560]
# Radford
df_r2 = get_obj_stats(base2, rv1)#.loc[lambda df: df['centroid-1'] < 550]
df_r3 = get_obj_stats(base3, rv2)#.loc[lambda df: df['centroid-1'] < 550]
df_rr = get_obj_stats(base_nc, rc1).loc[lambda df: df['centroid-1'] < 530]
df_rl = get_obj_stats(base_cl, rc2).loc[lambda df: df['centroid-0'] < 560]
df_rc = get_obj_stats(base_cv, rc3).loc[lambda df: df['centroid-0'] < 560]
# TRACE3D (Handwerker)
df_h2 = get_obj_stats2(base2, hv1)
df_h3 = get_obj_stats2(base3, hv2)
df_hr = get_obj_stats2(base_nc, hc1)
df_hl = get_obj_stats2(base_cl, hc2)
df_hc = get_obj_stats2(base_cv, hc3)
# tobac
df_t2 = get_obj_df(bv1)
df_t3 = get_obj_df(bv2)
df_tr = get_obj_df(bc1)
df_tl = get_obj_df(bc2)
df_tc = get_obj_df(bc3)
# convsf
df_c2 = get_obj_stats2(base2, cs_list[0])#.loc[lambda df: df['centroid-1'] < 550]
df_c3 = get_obj_stats2(base3, cs_list[1])#.loc[lambda df: df['centroid-1'] < 550]
df_cr = get_obj_stats2(base_nc, cs_list[2]).loc[lambda df: df['centroid-1'] < 530]
df_cl = get_obj_stats2(base_cl, cs_list[3]).loc[lambda df: df['centroid-0'] < 560]
df_cc = get_obj_stats2(base_cv, cs_list[4]).loc[lambda df: df['centroid-0'] < 560]
# artificial areas
M1 = get_obj_stats(base_gradient, gv1).area_filled.to_list()[0]
M2 = get_obj_stats(base_gradient, gv2).area_filled.to_list()[0]
MT1 = get_obj_stats(base_gradient, np.nanquantile(base2, 0.8, axis=None)).area_filled.to_list()[0]
MT2 = get_obj_stats(base_gradient, np.nanquantile(base3, 0.8, axis=None)).area_filled.to_list()[0]
#f3_over = base_nc.copy()
#f3_over[np.nonzero(base_gradient<gc1)]=float('nan')
f3_over = xr.where(base_gradient<gc1, float('nan'), base_nc)
M3 = get_obj_stats(f3_over, gc1).area_filled.to_list()[0]
t_u8 = np.nanquantile(base_nc, 0.8, axis=None)
tf3_over = xr.where(base_gradient<t_u8, float('nan'), base_nc)
MT3 = get_obj_stats(tf3_over, t_u8).area_filled.to_list()[0]
A1 = df_e2['area_filled'].sum() - M1
A2 = df_e3['area_filled'].sum() - M2
A3 = df_r['area_filled'].sum() - M3
A4 = df_l['area_filled'].sum()
A5 = df_c['area_filled'].sum()
B1 = df_r2['area_filled'].sum()
B2 = df_r3['area_filled'].sum()
B3 = df_rr['area_filled'].sum()
B4 = df_rl['area_filled'].sum()
B5 = df_rc['area_filled'].sum()
C1 = df_c2['area_filled'].sum()
C2 = df_c3['area_filled'].sum()
C3 = df_cr['area_filled'].sum()
C4 = df_cl['area_filled'].sum()
C5 = df_cc['area_filled'].sum()
H1 = df_h2['area_filled'].sum()
H2 = df_h3['area_filled'].sum()
H3 = df_hr['area_filled'].sum()
H4 = df_hl['area_filled'].sum()
H5 = df_hc['area_filled'].sum()
T1 = df_t2['area_filled'].sum() - MT1 + 568./3.+0.4*1698
T2 = df_t3['area_filled'].sum() - MT2 + 1117
T3 = df_tr['area_filled'].sum() - MT3
T4 = df_tl['area_filled'].sum() - 11351 # manually removed excess
T5 = df_tc['area_filled'].sum()
'''
input_list = [[0.750, 97, 97, 'ndi', None],
              [0.750, 100, 50, 'UQ', None],
              [0.833, 100, 50, 'US', 30]]

ct_sm = [] # regular ellipse counts
ct_mu = [] # core ellipse counts
ct_ell = [] # Laura irregular object counts
ct_irr = [] # irreular object counts
ct_low = [] # irregular weak object counts
ct_cv = [] # irregular convective object counts
area_sm = [] # regular ellipse total area
area_mu = [] # core ellipse total area
#area_ell = [] # Laura irregular object total area
area_irr = [] # irreular object total area
area_cl = [] # irregular weak object counts
area_cv = [] # irregular convective object counts

for inputs in input_list:
    if inputs[3] == 'ndi':
        fn = lambda arr: np.nanquantile(arr, inputs[0])
        tv1 = ndi.generic_filter(base2, function=fn, footprint=np.ones((inputs[1],inputs[1])), mode='nearest')
        print('tv1')
        tv2 = ndi.generic_filter(base3, function=fn, footprint=np.ones((inputs[1],inputs[1])), mode='nearest')
        print('tv2')
        tc1 = ndi.generic_filter(base_nc, function=fn, footprint=np.ones((inputs[1],inputs[1])), mode='nearest')
        tc2 = ndi.generic_filter(base_cl, function=fn, footprint=np.ones((inputs[1],inputs[1])), mode='nearest')
        tc3 = ndi.generic_filter(base_cv, function=fn, footprint=np.ones((inputs[1],inputs[1])), mode='nearest')
        sm_v1 = tv1 # do not use smoothed fields for scipy filter
        sm_v2 = tv2
        sm_c1 = tc1
        sm_c2 = tc2
        sm_c3 = tc3
    else:
        tv1, sm_v1 = rolling_window_arr(base2, *inputs[:3], max_thresh=inputs[-1])
        tv2, sm_v2 = rolling_window_arr(base3, *inputs[:3], max_thresh=inputs[-1])
        tc1, sm_c1 = rolling_window_arr(base_nc, *inputs[:3], max_thresh=inputs[-1])
        tc2, sm_c2 = rolling_window_arr(base_cl, *inputs[:3], max_thresh=inputs[-1])
        tc3, sm_c3 = rolling_window_arr(base_cv, *inputs[:3], max_thresh=inputs[-1])

    df_e2 = get_obj_stats(base2, tv1).loc[lambda df: df['centroid-1'] < 550]
    df_e3 = get_obj_stats(base3, tv2).loc[lambda df: df['centroid-1'] < 550]
    df_r = get_obj_stats(base_nc, tc1).loc[lambda df: df['centroid-1'] < 530]
    df_l = get_obj_stats(base_cl, tc2).loc[lambda df: df['centroid-0'] < 560]
    df_c = get_obj_stats(base_cv, tc3).loc[lambda df: df['centroid-0'] < 560]
    
    df_es2 = get_obj_stats(base2, sm_v1).loc[lambda df: df['centroid-1'] < 550]
    df_es3 = get_obj_stats(base3, sm_v2).loc[lambda df: df['centroid-1'] < 550]
    df_rs = get_obj_stats(base_nc, sm_c1).loc[lambda df: df['centroid-1'] < 530]
    df_ls = get_obj_stats(base_cl, sm_c2).loc[lambda df: df['centroid-0'] < 560]
    df_cs = get_obj_stats(base_cv, sm_c3).loc[lambda df: df['centroid-0'] < 560]

    # append counts and areas
    ct_sm.append([tv1, sm_v1])
    ct_mu.append([tv2, sm_v2])
    ct_irr.append([tc1, sm_c1])
    ct_low.append([tc2, sm_c2])
    ct_cv.append([tc3, sm_c3])
'''
    area_sm.append([df_e2['area_filled'].sum(), df_es2['area_filled'].sum()])
    area_mu.append([df_e3['area_filled'].sum(), df_es3['area_filled'].sum()])
    area_irr.append([df_r['area_filled'].sum(), df_rs['area_filled'].sum()])
    area_cl.append([df_l['area_filled'].sum(), df_ls['area_filled'].sum()])
    area_cv.append([df_c['area_filled'].sum(), df_cs['area_filled'].sum()])
'''
#ct_ell = np.asarray(ct_ell) # unsmoothed counts
ct_sm = np.asarray(ct_sm) # reg ellipse counts
ct_mu = np.asarray(ct_mu) # multi-core ellipse counts
ct_irr = np.asarray(ct_irr) # irreular object counts
ct_low = np.asarray(ct_low) # weak irreular object counts
ct_cv = np.asarray(ct_cv) # conv irreular object counts

# generate plot
figNameA = os.path.join(SAVEDIR, 'objects_synthetic_comparison_G18.png')
plot_by_alg(figNameA, [base2, base3, base_nc, base_cl, base_cv], [gv1, gv2, gc1, gc2, gc3])
figNameB = os.path.join(SAVEDIR, 'objects_synthetic_comparison_R19.png')
plot_by_alg(figNameB, [base2, base3, base_nc, base_cl, base_cv], [rv1, rv2, rc1, rc2, rc3])
figNameT = os.path.join(SAVEDIR, 'objects_synthetic_comparison_tobac.png')
plot_by_alg(figNameT, [base2, base3, base_nc, base_cl, base_cv],
            [bv1, bv2, bc1, bc2, bc3], stype='tobac')
figNameC = os.path.join(SAVEDIR, 'objects_synthetic_comparison_convsf.png')
plot_by_alg(figNameC, [base2, base3, base_nc, base_cl, base_cv], cs_list, stype='binary')
figNameN = os.path.join(SAVEDIR, 'objects_synthetic_comparison_alg-ndi.png')
plot_by_alg(figNameN, [base2, base3, base_nc, base_cl, base_cv],
            [ct_sm[0,0], ct_mu[0,0], ct_irr[0,0], ct_low[0,0], ct_cv[0,0]])
figNameQ = os.path.join(SAVEDIR, 'objects_synthetic_comparison_alg-UQ.png')
plot_by_alg(figNameQ, [base2, base3, base_nc, base_cl, base_cv],
            [ct_sm[1,1], ct_mu[1,1], ct_irr[1,1], ct_low[1,1], ct_cv[1,1]])
figNameS = os.path.join(SAVEDIR, 'objects_synthetic_comparison_alg-US30.png')
plot_by_alg(figNameS, [base2, base3, base_nc, base_cl, base_cv],
            [ct_sm[2,1], ct_mu[2,1], ct_irr[2,1], ct_low[2,1], ct_cv[2,1]])

#plot_objects(SAVEDIR, [base2, base3, base_nc, base_cl, base_cv],\
#             [gv1, gv2, gc1, gc2, gc3], [rv1, rv2, rc1, rc2, rc3],
#             [bv1, bv2, bc1, bc2, bc3],
#             cs_list, [ct_sm, ct_mu, ct_irr, ct_low, ct_cv], use_smooth=True)