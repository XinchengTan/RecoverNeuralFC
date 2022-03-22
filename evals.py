import numpy as np
from collections import namedtuple
from scipy.stats import pearsonr
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
from typing import List


FCRecoveryResult = namedtuple("FCRecoveryResult", ["auxi_sigma", "auxi_fitter", "gm_sigma", "gm_prec", "lam"])


class SingleRunResult(object):

  def __init__(self, fcRec_result: FCRecoveryResult, full_sigma, full_gm_sigma, full_gm_prec, full_gm_lam,
               edge_types, edge_cutoff_tol, edge_cutoff_qt, missing_entries_mask):
    self.fcRec_result = fcRec_result
    self.missing_entries_mask = np.array(missing_entries_mask, dtype=bool)
    self.simult_entries_mask = np.invert(self.missing_entries_mask)
    self.full_gm_lam = full_gm_lam
    self.full_sigma = full_sigma
    self.full_gm_sigma = full_gm_sigma
    self.full_gm_prec = full_gm_prec

    # Frob norm of matrix estimation error
    self.auxi_sigma_err = mat_rmse(fcRec_result.auxi_sigma, full_sigma)
    self.gm_sigma_err = mat_rmse(fcRec_result.gm_sigma, full_gm_sigma)
    self.gm_prec_err = mat_rmse(fcRec_result.gm_prec, full_gm_prec)

    # Matrix correlation
    self.auxi_sigma_corr = mat_corr(fcRec_result.auxi_sigma, full_sigma)
    self.gm_sigma_corr = mat_corr(fcRec_result.gm_sigma, full_gm_sigma)
    self.gm_prec_corr = mat_corr(fcRec_result.gm_prec, full_gm_prec)

    # Compute connectivity matrices
    self.full_A = prec_to_adjmat(full_gm_prec, standardize=True, qt=edge_cutoff_qt,
                                 tol=edge_cutoff_tol, edge_types=edge_types)
    self.Ahat = prec_to_adjmat(fcRec_result.gm_prec, standardize=True, qt=edge_cutoff_qt,
                                 tol=edge_cutoff_tol, edge_types=edge_types)
    # Accuracy, recall, precision, F-1 of connectivity graph
    self.acc = adj_mat_acc(self.full_A, self.Ahat)

    self.prec, self.recall, self.f1, _ = adj_mat_recall_prec_f1(self.full_A, self.Ahat)

  def __str__(self):
    single_result_str = ""
    for a in dir(self):
      if not a.startswith("__") and not callable(getattr(self, a)):
        single_result_str += "{:20} {}\n".format(a, getattr(self, a))
    return single_result_str


def unpack_result_list(results: List[SingleRunResult]):
  lams = []
  full_lams = []
  auxi_rmse, gm_sigma_rmse, gm_prec_rmse = [], [], []
  auxi_corr, gm_sigma_corr, gm_prec_corr = [], [], []
  acc, prec, recall, f1 = [], [], [], []
  adj_mats = []
  for result in results:
    lams.append(result.fcRec_result.lam)
    full_lams.append(result.full_gm_lam)
    auxi_rmse.append(result.auxi_sigma_err)
    auxi_corr.append(result.auxi_sigma_corr)
    gm_sigma_rmse.append(result.gm_sigma_err)
    gm_sigma_corr.append(result.gm_sigma_corr)
    gm_prec_rmse.append(result.gm_prec_err)
    gm_prec_corr.append(result.gm_prec_corr)
    acc.append(result.acc)
    prec.append(result.prec)
    recall.append(result.recall)
    f1.append(result.f1)
    adj_mats.append(result.Ahat)
  return lams, full_lams, auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
         acc, prec, recall, f1, adj_mats


def unpack_result_list_target_entries(results: List[SingleRunResult], target='missing'):
  auxi_rmses, gm_sigma_rmses, gm_prec_rmses = [], [], []
  auxi_corrs, gm_sigma_corrs, gm_prec_corrs = [], [], []
  accs, precs, recalls, f1s = [], [], [], []
  for result in results:
    target_entries_mask = result.missing_entries_mask if target == 'missing' else result.simult_entries_mask
    auxi_rmses.append(mat_rmse(result.fcRec_result.auxi_sigma, result.full_sigma, target_entries_mask))
    gm_sigma_rmses.append(mat_rmse(result.fcRec_result.gm_sigma, result.full_gm_sigma, target_entries_mask))
    gm_prec_rmses.append(mat_rmse(result.fcRec_result.gm_prec, result.full_gm_prec, target_entries_mask))
    auxi_corrs.append(mat_corr(result.fcRec_result.auxi_sigma, result.full_sigma,
                               target_entries_mask=target_entries_mask))
    gm_sigma_corrs.append(mat_corr(result.fcRec_result.gm_sigma, result.full_gm_sigma,
                                   target_entries_mask=target_entries_mask))
    gm_prec_corrs.append(mat_corr(result.fcRec_result.gm_prec, result.full_gm_prec,
                                  target_entries_mask=target_entries_mask))
    accs.append(adj_mat_acc(result.Ahat, result.full_A, target_entries_mask=target_entries_mask))
    prec, recall, f1, _ = adj_mat_recall_prec_f1(result.full_A, result.Ahat,
                                                 target_entries_mask=target_entries_mask)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1s)
  return auxi_rmses, auxi_corrs, gm_sigma_rmses, gm_sigma_corrs, gm_prec_rmses, gm_prec_corrs, accs, precs, recalls, f1s


def unpack_result_list_missing_entries(results: List[SingleRunResult]):
  return unpack_result_list_target_entries(results, target='missing')


def unpack_result_list_simult_entries(results: List[SingleRunResult]):
  return unpack_result_list_target_entries(results, target='simult')


def mat_rmse(X, Xhat, target_entries_mask=None):
  # Computes Frobenius norm of ||X - Xhat|| divided by p
  assert np.shape(X) == np.shape(Xhat), "Input matrices must have the same shape!"
  # RMSE on the full matrix
  if target_entries_mask is None:
    diff = np.abs(X - Xhat)
    return np.linalg.norm(diff, ord="fro") / np.shape(X)[0]

  # RMSE on the designated entries in X
  assert np.shape(X) == np.shape(target_entries_mask), 'Target entries mask must have the same shape as X, Xhat!'
  missing_entries_mask = np.array(target_entries_mask, dtype=bool)
  X = np.array(X[missing_entries_mask]).flatten()
  Xhat = np.array(Xhat[missing_entries_mask]).flatten()
  mse = mean_squared_error(X, Xhat)
  return np.sqrt(mse)


def mat_corr(A, B, standardize=True, square_symmetric=True, target_entries_mask=None):
  # Computes Pearson correlation between matrix A and B
  shapeA, shapeB = np.shape(A), np.shape(B)
  assert shapeA == shapeB, "Input matrices has different shapes: %s, %s" % (str(shapeA), str(shapeB))
  assert np.shape(target_entries_mask) == shapeA if target_entries_mask is not None else True, \
    "target entries mask must have the same shape as input matrices!"
  target_off_diag_idxs = get_target_off_diag_idxs(p=shapeA[0], target_entries_mask=target_entries_mask)

  if square_symmetric:
    #idxA, idxB = np.triu_indices(shapeA[0], 1), np.triu_indices(shapeB[0], 1)  # Only off-diagonal entries
    upperA, upperB = A[target_off_diag_idxs], B[target_off_diag_idxs]
    return pearsonr(upperA, upperB)[0]
  return pearsonr(np.matrix(A).flatten(), np.matrix(B).flatten())[0]


def standardize_prec(prec):
  p = len(prec)
  sqrt_diag = np.diag(prec) ** 0.5
  standardizer = np.reshape(sqrt_diag, (p, 1)) * np.reshape(sqrt_diag, (1, p))
  std_prec = prec / standardizer
  return std_prec


def prec_to_adjmat(prec, qt=0.05, tol=1e-8, standardize=True, edge_types=2):
  std_prec = standardize_prec(prec) if standardize else prec
  A = np.zeros_like(std_prec)
  P = np.abs(std_prec)
  if tol is None:
    tol = np.quantile(P[P != 0], qt)  # prec is flattened in the computation
  if edge_types == 3:
    A[std_prec > tol] = 1.0
    A[std_prec < -tol] = -1.0
  else:
    A[P > tol] = 1.0
  return A


def adj_mat_acc(A1, A2, target_entries_mask=None):
  # Exclude diagonal entries
  assert np.shape(A1) == np.shape(A2), "Input matrices must have the same shape!"
  assert np.shape(target_entries_mask) == np.shape(A1) if target_entries_mask is not None else True, \
    'target entries mask must have the same shape as the input matrices!'
  p = np.shape(A1)[0]
  target_off_diag_idxs = get_target_off_diag_idxs(p, target_entries_mask)
  upper1, upper2 = A1[target_off_diag_idxs], A2[target_off_diag_idxs]
  diff = np.abs(upper1 - upper2)
  return 1.0 - np.mean(diff)


def adj_mat_recall_prec_f1(A, Ahat, target_entries_mask=None):
  # Exclude diagonal entries
  assert np.shape(A) == np.shape(Ahat), "Input matrices must have the same shape!"
  assert np.shape(target_entries_mask) == np.shape(A) if target_entries_mask is not None else True, \
    'target entries mask must have the same shape as the input matrices!'
  p = np.shape(A)[0]
  target_off_diag_idxs = get_target_off_diag_idxs(p, target_entries_mask)
  upper1, upper2 = A[target_off_diag_idxs], Ahat[target_off_diag_idxs]
  return precision_recall_fscore_support(upper1.flatten(), upper2.flatten(), average="binary")


def get_target_off_diag_idxs(p, target_entries_mask):
  off_diag_idxs = np.triu_indices(p, 1)
  if target_entries_mask is None:
    # Entries of the full adjacency matrix
    return off_diag_idxs
  else:
    # Selected entries in the adjacency matrix
    target_off_diag_mask = target_entries_mask[off_diag_idxs]
    target_off_diag_idxs = (off_diag_idxs[0][target_off_diag_mask], off_diag_idxs[1][target_off_diag_mask])
    return target_off_diag_idxs


# def adj_acc(A, Ahat):
#   # Computes the percentage of matching entries of a binary adjacency matrix
#   diff = np.abs(A - Ahat)
#   return 1.0 - np.sum(diff) / np.size(A)
