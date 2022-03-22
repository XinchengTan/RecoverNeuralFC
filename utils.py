# Helper functions
from globals import *
from NearestPSD import shrinking

from collections import defaultdict
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.covariance import EmpiricalCovariance

import warnings


def get_missing_entries_mask(missing_fsp):
  return get_target_entries_mask(missing_fsp, target='missing')


def get_simult_entries_mask(missing_fsp):
  return get_target_entries_mask(missing_fsp, target='simult')


def get_target_entries_mask(missing_fsp, target='missing'):
  N = np.shape(missing_fsp)[0]
  target_entries_mask = np.zeros((N, N)) if target == 'missing' else np.ones((N, N))
  neuron_to_obs_timestamps = {}
  for neuron, fsp in enumerate(missing_fsp):
    neuron_to_obs_timestamps[neuron] = np.where(~np.isnan(fsp))[0]
  for neuron_i in range(N):
    for neuron_j in range(neuron_i+1, N):
      simult_timestamps = np.intersect1d(neuron_to_obs_timestamps[neuron_i], neuron_to_obs_timestamps[neuron_j])
      if len(simult_timestamps) == 0:  # Flip the init mask entry
        target_entries_mask[neuron_i][neuron_j] = 1 - target_entries_mask[neuron_i][neuron_j]
        target_entries_mask[neuron_j][neuron_i] = 1 - target_entries_mask[neuron_j][neuron_i]

  return target_entries_mask


def check_PD(mat, mat_name=""):
  # returns True if the input matrix is positive definite
  if mat.shape[0] != mat.shape[1]:
    raise ValueError("Input matrix for PD check must be a square matrix!")

  nan_entries = np.where(np.isnan(mat))
  inf_entries = np.where(np.isinf(mat))
  if len(nan_entries[0]) > 0:
    print("There are %d nan entries in mat!" % len(nan_entries[0]))
  if len(inf_entries[0]) > 0:
    print("There are %d inf entries in mat!" % len(inf_entries[0]))

  is_pd = np.all(np.linalg.eigvals(mat) > 0)
  chol_is_pd = shrinking.checkPD(mat)
  if is_pd and chol_is_pd:
    return True
  elif is_pd and not chol_is_pd:
    print("Checker inconsistent with Cholesky decomposition checker!")
    return None
  else:
    is_psd = np.all(np.linalg.eigvals(mat) >= 0)
    if not is_psd:
      print("%s Matrix is not positive semi-definite!" % mat_name)
  return False


def nan_percentage(mat):
  # return the percentage of nan entries
  nan_cnts = len(np.where(np.isnan(mat))[0])
  return nan_cnts / np.size(mat)


# Tuning curve of neuron responses to a tuning variable
def tuning_curve(fsp, tuning_var, bins=None, binrange=None, verbose=False):
  """

  :param fsp: Fluorescence trace with missing data
  :param tuning_var: target variable to tune against
  :param bins: number of ranges of the target variable to consider (None if var is categorical)
  :param binrange: range of bins (default to [min(tuning_var), max(tuning_var)])
  :param verbose:
  :return:
  """
  N, T = fsp.shape

  assert T == len(tuning_var), "Time axis of fsp and tuning variable does not align!"
  assert np.ndim(tuning_var) == 1 or (np.ndim(tuning_var) == 2 and np.shape(tuning_var)[1] == 1), \
    "Tuning variable should be 1-dimensional!"

  tuning_var = np.reshape(tuning_var, len(tuning_var))

  if bins is None:
    # Categorical variable
    var_vals = sorted(list(set(tuning_var)))
    avg_tuning_curve = np.zeros((len(var_vals), N))
    for vi, ival in enumerate(var_vals):
      fsp_ival = fsp[:, tuning_var == ival]
      avg_tuning_curve[vi] = np.nanmean(fsp_ival, axis=1)
    return var_vals, np.arange(len(var_vals)), avg_tuning_curve

  else:
    # If tuning_var is continuous, bin the values into evenly-spaced groups
    if binrange is None:
      binrange = (np.nanmin(tuning_var), np.nanmax(tuning_var)+0.000001)
    var_bins = np.linspace(binrange[0], binrange[1]+0.000001, bins)
    bins_cnt = len(var_bins) - 1
    avg_tuning_curve = []  # bins * N
    nonempty_bins = []
    for i in range(bins_cnt):
      fsp_ibin = fsp[:, (tuning_var >= var_bins[i]) & (tuning_var < var_bins[i + 1])]
      if fsp_ibin.shape[1] > 0:
        if not np.all(np.isnan(fsp_ibin)):  # Discard if all(fsp_ibin == np.nan)
          tc = np.nanmean(fsp_ibin, axis=1)
          avg_tuning_curve.append(tc)
          nonempty_bins.append(i)
        else:
          print("Discarding an all-NaN slice in fsp")
      # else:
      #   # Discard the bin if it has never occurred
      #   empty_bin.add(i)
    #var_bins = np.array([var_bins[i] for i in range(bins_cnt) if i not in empty_bin])
    #avg_tuning_curve = np.array([avg_tuning_curve[i, :] for i in range(bins_cnt) if i not in empty_bin])
    return var_bins, np.array(nonempty_bins), np.array(avg_tuning_curve)


# Correlation matrix of the tuning curve matrix across a neuron population
def tc_mat_corr(tc_mat, checkNan=True):
  # tc_mat has shape: (tuning_vars, neurons)
  tc_sim = np.array(pd.DataFrame(tc_mat).corr())  # pd.corr ignores nan

  if checkNan and np.any(np.isnan(tc_sim)):
    print("Tuning Curve Similarity Matrix contains NaN!")
  return tc_sim


def tc_cosine_sim(tc_mat, checkNan=True):
  # TODO: Compute cosine similarity of each pair of tuning curves
  TC, N = tc_mat.shape
  tc_norms = np.sqrt(np.nansum(tc_mat ** 2, axis=0))
  tc_cos_sim = np.identity(N)
  for i in range(N):
    for j in range(i+1, N):
      tc_cos_sim[i, j] = tc_cos_sim[j, i] = np.nansum(tc_mat[:, i] * tc_mat[:, j]) / (tc_norms[i] * tc_norms[j])

  if checkNan and np.any(np.isnan(tc_cos_sim)):
    print("Tuning Curve Cosine Similarity Matrix contains NaN!")
  return tc_cos_sim


def gaussian_k(xi, xj, sigma2):
  # exclude nan entries
  d = xi - xj
  return np.exp(-0.5 * np.nansum(d*d) / sigma2)


def tc_gaussian_kernel(tc_mat, sigma=0.001, checkNan=True):
  TC, N = tc_mat.shape
  tc_kernel_mat = np.identity(N)
  sigma2 = sigma ** 2
  for i in range(N):
    for j in range(i+1, N):
      tc_kernel_mat[i,j] = tc_kernel_mat[j,i] = gaussian_k(tc_mat[:,i], tc_mat[:,j], sigma2)

  if checkNan and np.any(np.isnan(tc_kernel_mat)):
    print("Tuning Curve Gaussian Kernel Matrix contains NaN!")
  return tc_kernel_mat


# Tuning curve correlation of two neurons
def tc_corr(tc1, tc2):
  return pearsonr(tc1, tc2)


def isSymmetric(X):
  return np.allclose(X, X.T, rtol=1e-8, atol=1e-8)


def isPSD(X):
  if X.ndim > 2:  return False
  if not isSymmetric(X):
    print("Warning: Input matrix is not symmetric!")
  eigenvals = np.linalg.eigvals(X)
  return all(eigenvals >= 0)


def isPD(X):
  if np.ndim(X) > 2: return False
  if not isSymmetric(X):
    print("Warning: Input matrix is not symmetric!")
  eigenvals = np.linalg.eigvals(X)
  return all(eigenvals > 0)



def newtons_PDCorrection(M0, *, M1=None, fbs=None, tol=10**(-6), maxIterations=None, checkM0=True):
  alpha = shrinking.newton(M0, M1=M1, fbs=fbs, tol=tol, maxIterations=maxIterations, checkM0=checkM0)
  pd_matrix = alpha * M1 + (1 - alpha) * M0
  return pd_matrix


# TODO: add ternary edge evals!!
