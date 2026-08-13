import tifffile
from numba import jit
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
#from scipy.stats import pearsonr
import os
from scipy.ndimage.morphology import binary_dilation
from skimage.measure import label, regionprops
from PIL import Image

@jit(nopython=True)
def pixels_in_disk_centered_at(i, j, i_max,j_max,radius):
    list_pixels = []
    r = math.floor(radius)
    for candidate_i in range(i-r,i+r+1):
        if candidate_i>=0 and candidate_i<i_max:
            for candidate_j in range(j-r,j+r+1):
                if candidate_j>=0 and candidate_j<j_max:
                    distance = np.sqrt((i-candidate_i)**2+(j-candidate_j)**2)
                    if distance <= r:
                        list_pixels.append((candidate_i, candidate_j))
    return list_pixels

@jit(nopython=True)
def corr_disk_centered_at(i,j,stack,radius):
    T,imax,jmax = stack.shape
    vec_ij = stack[:,i,j]
    centered_disk = pixels_in_disk_centered_at(i, j, imax,jmax,radius)
    correlations = []
    for kl in centered_disk:
        if kl!=(i,j):
            k,l = kl
            vec_kl = stack[:,k,l]
            corr= np.corrcoef(vec_ij, vec_kl)
            corr = corr[0,1]
            correlations.append(corr)
    correlation_array = np.array(correlations)
    mean_corr = np.mean(correlation_array)
    return mean_corr

@jit(nopython=True)
def corr_disk_stack(stack,radius):
    T,imax,jmax = stack.shape
    corr_mat = np.zeros((imax,jmax))
    for i in range(0,imax):
        for j in range(0,jmax):
            corr_ij = corr_disk_centered_at(i,j,stack,radius)
            corr_mat[i,j] = corr_ij
    return corr_mat


def mat_disk(radius):
    r = math.floor(radius)
    N = 2*r+1
    P = pixels_in_disk_centered_at(math.floor(N/2),math.floor(N/2),N,N,r)
    A = np.zeros((N,N))
    for ij in P:
        A[ij] = 1
    return A

def find_peaks(image, min_distance = 25, mean_thr = 0.8, display = True):
    """
    min_distance = positive integer, minimum number of pixel separating peaks
    mean_thr = positive number, if mean_thr = 0.5, then all peaks with values above 0.5 mean are accepted
    """
    # find peaks
    peak_coordinates = peak_local_max(image, min_distance)

    # get values at peaks
    peak_values = []
    for ij in peak_coordinates:
        peak_values.append(image[ij[0],ij[1]])
    mean_val = np.mean(peak_values)

    # filter peaks using the mean value
    selected_coordinates=[]
    for ij in peak_coordinates:
        if image[ij[0],ij[1]]> mean_thr*mean_val:
            selected_coordinates.append((ij[0],ij[1]))

    N_selected = len(selected_coordinates)

    array_coordinates = np.zeros((N_selected,2))
    for i in range(N_selected):
        array_coordinates[i,0] = selected_coordinates[i][0]
        array_coordinates[i,1] = selected_coordinates[i][1]

    if display:
        # display results
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].imshow(image, cmap=plt.cm.gray)
        ax[1].autoscale(False)
        ax[1].plot(array_coordinates[:, 1], array_coordinates[:, 0], 'r.')
        ax[1].axis('off')
        ax[1].set_title('Peak local max')

        fig.tight_layout()

        plt.show()

    return array_coordinates

def segment_with_peaks(image, radius = 13, mean_thr = 0.9, display = True):
    min_separation_distance = 2*radius

    peak_coordinates = find_peaks(image, min_separation_distance, mean_thr, display)

    mask = np.zeros(image.shape)
    for ij in peak_coordinates:
        mask[int(ij[0]),int(ij[1])] = 1.0

    small_radius = math.ceil(radius/2)
    disk = mat_disk(small_radius)
    dilated_mask = binary_dilation(mask, structure=disk)

    labeled_mask = label(dilated_mask)

    return labeled_mask

def get_time_series(stack,labeled_mask):
    nbr_neurons = np.max(labeled_mask.flatten())
    T, ni, nj = stack.shape
    time_series = np.zeros((nbr_neurons, T))
    for neuron in range(nbr_neurons):
        neuron_mask = labeled_mask==(neuron+1)
        nbr_pixels = np.sum(neuron_mask)
        for t in range(T):
            frame = stack[t,:,:].reshape(ni,nj)
            neuron_activities = frame[neuron_mask]
            mean_activity = np.sum(neuron_activities)/nbr_pixels
            time_series[neuron,t] = mean_activity

    mean_stack_std = np.mean(np.std(stack,axis=0))
    std_per_neuron = np.std(time_series,axis=1)
    active_neurons = []
    for i in range(nbr_neurons):
        if std_per_neuron[i]>mean_stack_std:
            active_neurons.append(i)

    return time_series, active_neurons


def normalize_time_series(time_series):
    N,T = time_series.shape
    normalized_time_series = np.zeros((N,T))
    for i in range(N):
        activity = time_series[i,:]-np.min(time_series[i,:])
        max_activity = np.max(activity)
        if max_activity>0:
            normalized_time_series[i,:] = activity/max_activity
    return normalized_time_series

def get_centroids(labeled_mask):
    props = regionprops(labeled_mask)
    N = len(props)
    positions = {}
    for neuron in range(N):
        ij = props[neuron].centroid
        x = int(ij[1])
        y = int(ij[0])
        positions[neuron] = (x,y)

    return positions
