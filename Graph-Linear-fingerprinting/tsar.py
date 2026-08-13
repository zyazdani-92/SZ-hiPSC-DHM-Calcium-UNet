#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short Python module, part of the Time Series Repository (TSAR)
@author: pdesrosiers
"""


import numpy as np
import math
from scipy import stats
from scipy.stats import pearsonr
from scipy.stats import zscore
from scipy.signal import coherence, hilbert
from scipy.special import betainc
from numba import jit

def normalize_data(data, method = "standard"):
    """
    Normalize each row of a matrix.

    Parameters
    ----------
    data : 2d array of size m x n
            data matrix for m variables and n sample points
    method : string, the method used for normalization

    Returns
    -------
    normalized_data : normalized 2d array of size m x n
    """
    if method == "zscore":
        normalized_data  = zscore(data, axis = 1)
    elif method == "standard":
        m,n = np.shape(data)
        min_vec = data.min(axis=1).reshape(m,1)
        normalized_data = data-min_vec
        max_vec = normalized_data.max(axis=1).reshape(m,1)
        normalized_data = normalized_data/max_vec
    return normalized_data

def energy(data):
    """
    Compute the energy of each row of a matrix and return statistics about the set of energies.

    Parameters
    ----------
    data : 2d array of size m x n
            data matrix for m variables and n sample points

    Returns
    -------
    energy_stats :  dictionary with keys 'min', 'max', 'mean', 'median', 'mode',
                    'std', 'mad', 'skewness', 'energy_vec'
    """

    energy_vec = np.sum(data**2, axis=1)
    energy_stats = basic_stats(energy_vec)
    energy_stats['energy_vec'] = energy_vec

    return energy_stats

def kinetic_energy(data):
    """
    Compute the kinetic energy (i.e. the energy of diffrences) of each row of a
    matrix and return statistics about the set of energies.

    Parameters
    ----------
    data : 2d array of size m x n
            data matrix for m variables and n sample points

    Returns
    -------
    energy_stats :  dictionary with keys 'min', 'max', 'mean', 'median', 'mode',
                    'std', 'mad', 'skewness', 'energy_vec'
    """

    differences = np.diff(data, n=1, axis=1)
    energy_stats = energy(differences)

    return energy_stats


def basic_stats(data):
    """
    Compute basics statistics about a sequence of data points.

    Parameters
    ----------
    data : 1d array

    Returns
    -------
    stat_dict : dictionary with keys 'min', 'max', 'mean', 'median', 'mode',
                'std', 'mad', 'skewness'
    """
    stat_dict = {}
    stat_dict['min'] = np.min(data)
    stat_dict['max'] = np.max(data)
    stat_dict['mean'] = np.mean(data)
    stat_dict['median'] = np.median(data)
    stat_dict['std'] = np.std(data)
    stat_dict['mode'] = stats.mode(data, axis = None).mode[0]
    stat_dict['mad'] = stats.median_absolute_deviation(data, axis = None)
    stat_dict['skewness'] = stats.skew(data, axis=None)
    return stat_dict


def pvalues_corr(r, n):
    """
    Computes the p-values of a correlation matrix using an asymptotic formula.

    Parameters
    ----------
    r : 2d array
        correlation matrix
    n : positive integer
        sample size

    Returns
    -------
    pval: 2d array of pvalues for the correlation matrix r
    """
    # return
    # Degrees of freedom
    df = (n-2)*np.ones(np.shape(r))
    # t-squared variable
    t_squared = np.divide(r*r*df, np.ones(np.shape(r))-r*r+1e-16) #1e-16 added to avoid division by zero
    # new variable
    new = np.divide(df, df + t_squared)
    np.where(new<0.0,0.0,new)
    np.where(new>1.0,1.0,new)
    # pvalue
    pval = betainc(0.5*df, 0.5*np.ones(np.shape(r)), new);
    return pval

def exact_pvalues_corr(corr, data, N):
    """
    Computes the p-values of a correlation matrix using permutations (exact test).

    Parameters
    ----------
    corr : 2d array m x m
            computed correlation matrix
    data : 2d array of size m x n
            data matrix for m variables and n sample points
    N : positive integer
        number of permutations

    Returns
    -------
    pval: 2d array with p-values
    """

    m,n = np.shape(data)
    copy_data = np.copy(data)
    abs_r = np.abs(corr)
    counts = np.zeros((m,m))

    for i in range(0,N):
        new_r = np.abs(np.corrcoef(shuffle_in_each_row(copy_data)))
        counts[np.where(new_r>abs_r)]+=1

    return counts/N


def shuffle_in_each_row(data):
    mat = np.copy(data)
    m,n = np.shape(mat)
    for i in range(0,m):
        np.random.shuffle(mat[i,:])
    return mat

def significant_corr(data, alpha = 0.01, exact = False):
    """
    Parameters
    ----------
    data : 2d array of size mx n
            data matrix for m variables and n sample points
    alpha : significance level
            real between 0 and 1 (default 0.01)
    exact : boolean
            if true, the exact p-values are computed by permuting the data points in each time series
            if false, the p-values are computed with the asymptotic formula

    Returns
    -------
    pearson : 2d array of significant correlations
    """
    m,n = np.shape(data)
    r = np.corrcoef(data)
    if exact:
        N = int(np.round(1/alpha*10))
        p = exact_pvalues_corr(r, data, N)
    else:
        p = pvalues_corr(r, n)

    return np.where(p>alpha,0.0,r)


#@jit(nopython=True)
def significant_xcorr(data, alpha = 0.01, max_lag = 5):
    """
    Compute all significant maximum cross correlations between all pairs of variables.
    The p-values are computed with the asymptotic formula

    Parameters
    ----------
    data :   2d array of size mx n
             data matrix for m variables and n sample points
    alpha :  significance level
             real between 0 and 1 (default 0.01)
    max_lag :non neg integer, number of time steps used
             in the past to compute the cross-correlations

    Returns
    -------
    xr :     2d array of significant cross-correlations
             the element (i,j) is the correlation of the past of the j-th variable
             with the present of the i-th variable; it is a proxy for the measure
             of causality from j to i

    lags :   2d array of time lags with maximum absolute correlation
             the element (i,j) provides the time lag t for which the past of variable j
             is maximally correlated with the present of vraible i
    """
    m,n = np.shape(data)
    xr = np.zeros((m,m))
    lags = np.zeros((m,m))
    for j in range(m):
        for i in range(m):
            xr_ji_vec = np.zeros((max_lag,))
            for t in range(1,max_lag+1):
                idata = data[i,t:n]
                jdata = data[j,0:n-t]
                xr_ji_vec[t-1] = np.corrcoef(idata,jdata)[0,1]

            xr_ji_vec_abs = np.abs(xr_ji_vec)# absolute value of corr coeff
            lag_ji = np.where(xr_ji_vec_abs == np.max(xr_ji_vec_abs))[0][0] #find position for max abs corr
            lags[j,i] = lag_ji +1
            xr[j,i] = xr_ji_vec[lag_ji]

    p = pvalues_corr(xr, n)

    return np.where(p>alpha,0.0,xr), lags


def srank(data):
    """
    Compute the stable rank of the data matrix.

    Parameters
    ----------
    data :  2d array of size m x n
            data matrix for m variables (features) & n sample points (time steps)

    Returns
    -------
    sr :  float, nonnegative, stable rank of the data matrix

    References
    ----------
    Magen, Avner, and Anastasios Zouzias. "Low rank matrix-valued Chernoff bounds and approximate matrix multiplication."
    Proceedings of the twenty-second annual ACM-SIAM symposium on Discrete Algorithms. SIAM, 2011.

    """
    u, s, vh = np.linalg.svd(data, full_matrices=True)
    Frobenius_norm_squared = np.sum(s**2)
    Spectral_norm_squared = s[0]**2
    if Spectral_norm_squared>0:
        sr = Frobenius_norm_squared/Spectral_norm_squared
    elif Spectral_norm_squared==0:
        sr = 0
    return sr

def rank(data):
    """
    Return the matrix rank of a data matrix using the classical numpy function.
    """
    return np.linalg.matrix_rank(data)

def Omega_complexity(data, normalized = True):
    """
    Computes the omega complexity index and its normalized version

    Parameters
    ----------
    data : 2d array of size mx n
            data matrix for m variables and n sample points
    normalized :    boolean
                    If true, computes the normalized version which varies
                    between 1/n (min synch) and 1 (max synch)
    Returns
    -------
    1/Omega (normalized) or Omega

    References
    -------
    [1] Wackermann, J. (1995). Beyond mapping: estimating complexity
         [...], Acta neurobiologiae experimentalis, 56(1), 197-208.
    [2] Jalili, M., Barzegaran, E., & Knyazeva, M. G. (2014).
        Synchronization of EEG: Bivariate and multivariate measures.
        IEEE Transactions on, 22(2), 212-221."""
    Corr = np.corrcoef(data)
    Eig = np.linalg.eigvalsh(Corr)
    Eig = np.abs(Eig) # accidental negatives may occur, so they are removed
    P = Eig/len(Eig)
    logP = np.nan_to_num(np.log(P))
    Omega = np.exp(np.sum(-P*logP))
    if normalized:
        return 1/Omega
    else:
        return Omega

def multivar_coherence(data):
    """
    Computes the magnitude squared coherence between pairs of time series,
    which is computed using the Fourier power spectrum of time series.

    Parameters
    ----------
    data : 2d array of size mx n
            data matrix for m variables and n sample points

    Returns
    -------
    coh_avg :   real between 0 (no coherence) and 1 (perfect coherence)
                average coherence between all pairs of times series
    coh_mat :   2d array
                element (i,j) is the mean (over frequencies) pf
                magnitude squared coherence between var i and j, which
                is  real between 0 (no synch) and 1 (perfect synch)

    References
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html
    """
    m,n = np.shape(data)
    coh_mat = np.zeros((m,m))
    for i in range(0,m-1):
        for j in range(i+1,m):
            freq, coh = coherence(data[i,:], data[j,:], nperseg = int(np.round(n/8)))
            coh_mat[i,j] = np.mean(coh)
    coh_avg = np.mean(coh_mat[np.nonzero(coh_mat)])
    coh_mat = coh_mat + np.transpose(coh_mat) + np.eye(m)
    return coh_avg, coh_mat

def phase_locking(data):
    """
    Computes the multivariate and the birariate phase locking values (PLV) for
    a set of time series. PLV indicate how a system is phase synchronized.

    Parameters
    ----------
    data :          2d array of size m x n
                    data matrix for m variables and n sample points

    Returns
    -------
    multivarPLV :   float
                    multivariate PLV between 0 (no synch) and 1 (perfect synch)
    bivarPLV :      mxm array whose elements are the bivariate PLV
    """
    m,n = np.shape(data)
    avg = np.mean(data, axis=1).reshape(m,1)
    centered_data = data - avg
    complex_signal = hilbert(centered_data, axis = 1)
    phases = np.angle(complex_signal)
    multivarPLV = np.mean(np.abs(np.mean(np.exp(1j*phases), axis =0)))
    bivarPLV = np.zeros((m,m))
    for i in range(0,m-1):
        for j in range(i+1,m):
            bivarPLV[i,j] = np.abs(np.sum(np.exp(1j*phases[i,:]-1j*phases[j,:])))/n
    bivarPLV = bivarPLV + np.transpose(bivarPLV) + np.eye(m)
    return multivarPLV, bivarPLV


def spectral_entropy(data, units = 'bit'):
    """
    Compute the Shannon entropy associated to the singular value decomposition
    of a data matrix. This entropy is often called spectral entropy. If the data
    matrix is a correlattion matrix, then the spectral entropy is related to
    the Omega complexity measure.

    Parameters
    ----------
    data :  2d array of size m x n
            data matrix for m variables (features) & n sample points (time steps)
    units : string, either 'bit' or 'nat', the unit of information

    Returns
    -------
    ent :  float, Shannon entropy either in bits or in nats

    References
    ----------
    W. Yang, J.D. Gibson, and T. He,'Coefficient rate and lossy source coding',
    IEEE transactions on information theory 51.1 (2005): 381-386.

    """
    u, s, vh = np.linalg.svd(data, full_matrices=True)
    sum_s = np.sum(s)
    prob_vec = s/sum_s
    ent = entropy_from_prob(prob_vec, units)

    return ent

def entropy_from_prob(vector, units = 'bit') :
    """
    Compute the Shannon entropy of a probability vector.

    """
    probability_vector = np.array(vector)
    positivity  =  np.prod((probability_vector>=0).astype(int) )
    normalization  = np.sum(probability_vector)

    if positivity!=1 :
        raise Exception("The argument must be a vector with nonnegative elements.")
    elif not math.isclose(normalization,1) :
        print(normalization)
        raise Exception("The elements of the argument must add to 1.")

    if units == 'bit':
        infos = -np.nan_to_num(np.log2(probability_vector))
        ent = np.sum(probability_vector*infos)
    elif units == 'nat':
        infos = -np.nan_to_num(np.log(probability_vector))
        ent = np.sum(probability_vector*infos)

    return ent

def erank(data):
    """
    Compute the Roy-Vetterli effective rank based on the distribution of
    singular values.

    Parameters
    ----------
    data :  2d array of size m x n
            data matrix for m variables (features) & n sample points (time steps)

    Returns
    -------
    e_rank :    float, the estimaded rank (dimension) of the data

    References
    ----------
    O. Roy & and M. Vetterli, 'The effective rank: A measure of effective
    dimensionality', 2007 15th European Signal Processing Conference. IEEE, 2007.
    """
    ent = spectral_entropy(data, units = 'nat')
    e_rank = np.exp( ent )

    return e_rank

def zero_crossing_rate(data):
    """
    For each row, compute the proportion of sign changes.

    Parameters
    ----------
    data : 2d array of size m x n
            data matrix for m variables (features) and n sample points (time steps)

    Returns
    -------
    zcr_stats : dictionary, contains statistics about the distribution of the
                zero crossing rates
    """

    zscores = normalize_data(data, 'zscore')
    m,n =  zscores.shape

    zcr_vec = np.zeros((m,1))
    for i in range(0,m):
        zcr_vec[i,0] = np.sum((zscores[i,0:n-1]*zscores[i,1:n]<0).astype('int'))/(n-1)

    zcr_stats = basic_stats(zcr_vec)
    zcr_stats['zcr_vec'] = zcr_vec
    return zcr_stats
