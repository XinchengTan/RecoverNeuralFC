import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

from evals import standardize_prec
from evals import unpack_result_list, unpack_result_list_missing_entries, unpack_result_list_simult_entries


def display_missing_fsp(missing_fsp, xlabel="Timepoints", ylabel="Neurons", title="Neuron Trace with Missing Entries"):
  X = np.nan_to_num(missing_fsp, nan=-9999)  # set nan entries to -9999
  ax = sns.heatmap(X, cmap="plasma")
  ax.set_xlabel(xlabel, fontsize=14)
  ax.set_ylabel(ylabel, fontsize=14)
  ax.set_title(title, fontsize=18, y=1.006)
  plt.savefig("./partialX.png")
  plt.show()


def display_missing_cov(missing_cov, cmap="viridis"):
  X = np.nan_to_num(missing_cov, nan=-9999)
  ax = sns.heatmap(X, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  ax.set_title("Covariance of a Neuron Trace with Missing Entries")
  plt.show()


def display_eigenvals(X, value_range=(float("-inf"), float("inf"))):
  low, high = value_range[0], value_range[1]
  eigvals = np.linalg.eigvals(X)
  plot_vals = eigvals[(low < eigvals) & (eigvals <= high)]

  # Plot the eigenvalue distribution
  plt.hist(plot_vals, bins=50, color="purple", alpha=0.75)
  plt.title("Eigenvalue Distribution")
  plt.xlabel("Eigenvalues")
  plt.ylabel("Number of eigenvalues")
  plt.show()


# Plot a pair-wise matrix via heatmap
def plot_matrix(mat, title="", cmap="BuPu", figsize=(10, 7)):
  fig, ax = plt.subplots(figsize=figsize)
  if not title:
    title = "Matrix of %d neurons" % mat.shape[0]
  ax.set_title(title, fontsize=16)
  sns.heatmap(mat, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  plt.show()


# Plot precision matrix
def plot_prec(prec, alpha, ax=None, standardize=True, label="", cmap="viridis"):
  P = np.array(prec)
  if standardize:
    P = standardize_prec(prec)
  if ax:
    sns.heatmap(P, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(P, cmap=cmap)
  ax.set_xlabel("Neurons")
  ax.set_ylabel("Neurons")
  ax.set_title(r"Precision Matrix [%s, $\lambda$ = %.2f]" % (label, alpha))
  plt.show()


# Plot adjacency matrix
def plot_adj_mat(A, ax=None, label="", include_negs=False):
  plt.figure(figsize=(12, 9))
  A2 = A * 3000
  cmap = "cividis" if not include_negs else "bwr"
  if ax:
    sns.heatmap(A2, cmap=cmap, ax=ax)
  else:
    ax = sns.heatmap(A2, cmap=cmap)
  ax.set_xlabel("Neurons", fontsize=18)
  ax.set_ylabel("Neurons", fontsize=18)
  total_edges = len(A[A != 0])
  if not include_negs:
    ax.set_title("Adjacency Matrix [%s] [%d edges]" % (label, total_edges), fontsize=20, y=1.05)
  else:
    pos_edges = len(A[A > 0])
    neg_edges = len(A[A < 0])
    ax.set_title("Adjacency Matrix [%s] [%d edges: %d+, %d-]" % (label, total_edges, pos_edges, neg_edges), y=1.05)
  plt.show()


# Plot connectivity graph based on an adjacency matrix and a location matrix
def plot_connectivity_graph(A, xys, cmap=plt.cm.coolwarm, ax=None, label=""):
  plt.figure(figsize=(12, 9))

  G = nx.convert_matrix.from_numpy_array(A)
  colors = [A[i][j] for i, j in G.edges]
  nx.draw_networkx(G, pos=xys, node_color="orange", alpha=0.85,
                   width=2, edge_cmap=cmap, edge_color=colors)
  plt.axis('equal')


def plot_fitted_lr(X, y, reg, cov):
  plt.scatter(X, y, color="red", alpha=0.7)
  plt.plot(X, reg.predict(X), color="blue")
  plt.title("Fitted Linear Regression")
  plt.xlabel("Auxiliary Information")
  plt.ylabel("Covariance") if cov else plt.ylabel("Correlation")
  plt.show()


def display_results_diff_lambdas(lambdas, session2res, varIsLam=True):
  fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(20, 35))
  xlbl = r"$\lambda$" if varIsLam else r"$\sigma$"

  for session, res in session2res.items():
    neuron_cnts = res.get("neuron_count")
    label = session + " (%d neurons)" % (neuron_cnts)
    results = res.get("results")
    lams, full_lams, auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
    acc, prec, recall, f1, adj_mats = unpack_result_list(results)
    print(label, " selected lamda on full data: ", full_lams)
    ax1.plot(lambdas, auxi_rmse, "o-", alpha=0.8, label=label)
    ax1.set_title(r"$||S - \hat{\Sigma}||_F$", fontweight="bold", fontsize=18)
    ax1.set_xlabel(xlbl, fontsize=14)
    ax1.set_ylabel("RMSE")
    ax1.legend(fontsize=14)

    ax2.plot(lambdas, auxi_corr, "o-", alpha=0.8, label=label)
    ax2.set_title("Correlation of $S, \hat{\Sigma}$", fontweight="bold", fontsize=18)
    ax2.set_xlabel(xlbl, fontsize=14)
    ax2.set_ylabel("Matrix correlation")

    ax3.plot(lambdas, gm_sigma_rmse, "o-", alpha=0.8, label=label)
    ax3.set_title(r"$||\hat{\Sigma}_{GM(S, \lambda)} - \hat{\Sigma}_{GM(\hat{\Sigma}, \lambda)}||_F$",
                  fontweight="bold", fontsize=18)
    ax3.set_xlabel(xlbl, fontsize=14)
    ax3.set_ylabel("RMSE")

    ax4.plot(lambdas, gm_sigma_corr, "o-", alpha=0.8, label=session + " (" + str(neuron_cnts) + " neurons)")
    ax4.set_title("Correlation of $\hat{\Sigma}_{GM(S, \lambda)} - \hat{\Sigma}_{GM(\hat{\Sigma}, \lambda)}$",
                  fontweight="bold", fontsize=18)
    ax4.set_xlabel(xlbl, fontsize=14)
    ax4.set_ylabel("Matrix correlation")

    ax5.plot(lambdas, gm_prec_rmse, "o-", alpha=0.8, label=label)
    ax5.set_title(r"RMSE of $||\hat{\Theta}_{full} - \hat{\Theta}||$", fontweight="bold", fontsize=18)
    ax5.set_xlabel(xlbl, fontsize=14)
    ax5.set_ylabel("RMSE", fontsize=14)

    ax6.plot(lambdas, gm_prec_corr, "o-", alpha=0.8, label=label)
    ax6.set_title(r"Off-diagonal Correlation of $\hat{\Theta}_{full}, \hat{\Theta}$", fontweight="bold", fontsize=18)
    ax6.set_xlabel(xlbl, fontsize=14)
    ax6.set_ylabel("Matrix correlation", fontsize=14)

    ax7.plot(lambdas, acc, "o-", label=label)
    ax7.set_title("Edge Prediction Accuracy", fontweight="bold", fontsize=18,y=1.005)
    ax7.set_xlabel(xlbl, fontsize=14)
    ax7.set_ylabel("Accuracy")

    ax8.plot(lambdas, prec, "o-", label=label)
    ax8.set_title("Edge Prediction Recall", fontweight="bold", fontsize=18,y=1.005)
    ax8.set_xlabel(xlbl, fontsize=14)
    ax8.set_ylabel("Recall")
    ax8.legend(fontsize=13)

    # TODO: bug fixed in evals!
    if "M15" in session:
      recall[-1] = np.nan
    ax9.plot(lambdas, recall, "o-", label=label)
    ax9.set_title("Edge Prediction Precision", fontweight="bold", fontsize=18,y=1.005)
    ax9.set_xlabel(xlbl, fontsize=14)
    ax9.set_ylabel("Precision")


    ax10.plot(lambdas, f1, "o-", label=label)
    ax10.set_title("Edge Prediction F1 Score", fontweight="bold", fontsize=18,y=1.005)
    ax10.set_xlabel(xlbl, fontsize=14)
    ax10.set_ylabel("F1 Score")

  plt.subplots_adjust(hspace=0.23)
  plt.savefig("diff_lambdas.svg")
  plt.show()


def set_title_xticklabels_ylabel_ylim(ax, title, xs, xticklabels, ylabel, ylim=[0, 1.1]):
  ax.set_title(title, fontweight="bold", fontsize=18, y=1.005)
  ax.set_xticks(xs)
  ax.set_xticklabels(xticklabels, rotation=50)
  ax.set_ylabel(ylabel, fontsize=13)
  ax.set_ylim(ylim)
  ax.legend()


def display_results_diff_auxi_show_all_sessions(auxi_labels, session2res):
  accs, precs, recalls, f1s = [], [], [], []
  auxi_rmses, auxi_corrs = [], []
  fig, ((ax1), (ax2), (ax3), (ax4), (ax5), (ax6)) = plt.subplots(6, 1, figsize=(15, 43))
  xs = np.arange(len(auxi_labels))
  for session, res in session2res.items():
    neuron_cnts = res.get("neuron_count")
    label = session + " (%d neurons)" % (neuron_cnts)
    results = res.get("results")
    lams, full_lams, auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
      acc, prec, recall, f1, adj_mats = unpack_result_list(results)

    accs.append(acc)
    precs.append(prec)
    recalls.append(recall)
    f1s.append(f1)

    auxi_rmses.append(auxi_rmse)
    auxi_corrs.append(auxi_corr)

    # Output gm estimation for full data
    opt_lam_full = res.get("full_gm_lam")
    print("Optimal Lambda for full data: ", opt_lam_full)
    # plot_adj_mat(A=results[0].full_A, label=label + "--full data")

    # Plot Acc, precision, recall, F1 for each session
    ax1.plot(xs, accs[-1], linestyle=':', marker='^', label=session)
    ax2.plot(xs, precs[-1], linestyle=':', marker='^', label=session)
    ax3.plot(xs, recalls[-1], linestyle=':', marker='^', label=session)
    ax4.plot(xs, f1s[-1], linestyle=':', marker='^', label=session)
    ax5.plot(xs, auxi_rmses[-1], linestyle=':', marker='.', label=session)
    ax6.plot(xs, auxi_corrs[-1], linestyle=':', marker='.', label=session)
    set_title_xticklabels_ylabel_ylim(ax1, "Accuracy of Recovered Adjacency Matrix",
                                      xs-0.3, auxi_labels, 'Accuracy', ylim=[0.5, 1.1])
    set_title_xticklabels_ylabel_ylim(ax2, "Precision of Recovered Adjacency Matrix",
                                      xs-0.3, auxi_labels, 'Precision')
    set_title_xticklabels_ylabel_ylim(ax3, "Recall of Recovered Adjacency Matrix",
                                      xs-0.3, auxi_labels, 'Recall')
    set_title_xticklabels_ylabel_ylim(ax4, "F1 Score of Recovered Adjacency Matrix",
                                      xs-0.3, auxi_labels, 'F1 Score')
    set_title_xticklabels_ylabel_ylim(ax5, "Sample Covariance RMSE",
                                      xs, auxi_labels, 'RMSE', ylim=[0, 0.1])
    set_title_xticklabels_ylabel_ylim(ax6, r"Off-diagonal Correlation of $S, \hat{S}$",
                                      xs, auxi_labels, 'Matrix correlation')
    plt.subplots_adjust(hspace=0.6)
  return


# Show evaluation of different combinations of auxiliary information
def display_results_diff_auxi(auxi_labels, session2res, target=None):

  indx = range(len(auxi_labels))
  if len(auxi_labels) == 6:
    my_colors = ["C7", "C0", "C9", "C1", "C8", "C3"]
  elif len(auxi_labels) == 11:
    my_colors = ["C7", "C0", "C9", "C1", "C8", "C0", "C9", "C1", "C8",
                 "C4", "C3"]
  else: #lif len(auxi_labels) == 16:
    my_colors = ["C7",
                 "C0", "C9", "C1", "C8", "C0", "C9", "C1", "C8", "C0", "C9", "C1", "C8",
                 "C6", "C4", "C3"]

  avg_acc, avg_prec, avg_recall, avg_f1 = [], [], [], []
  avg_auxi_rmse, avg_auxi_corr = [], []

  for session, res in session2res.items():
    neuron_cnts = res.get("neuron_count")
    label = session + " (%d neurons)" % (neuron_cnts)
    results = res.get("results")
    if target is None:
      lams, full_lams, auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
        acc, prec, recall, f1, adj_mats = unpack_result_list(results)
    elif target == 'missing':
      auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
        acc, prec, recall, f1 = unpack_result_list_missing_entries(results)
    else:
      auxi_rmse, auxi_corr, gm_sigma_rmse, gm_sigma_corr, gm_prec_rmse, gm_prec_corr, \
        acc, prec, recall, f1 = unpack_result_list_simult_entries(results)

    avg_acc.append(acc)
    avg_prec.append(prec)
    avg_recall.append(recall)
    avg_f1.append(f1)

    # Output gm estimation for full data
    opt_lam_full = res.get("full_gm_lam")
    print("Optimal Lambda for full data: ", opt_lam_full)
    #plot_adj_mat(A=results[0].full_A, label=label + "--full data")

    avg_auxi_rmse.append(auxi_rmse)
    avg_auxi_corr.append(auxi_corr)

    # fig, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(22, 16))
    # #ax3.bar(indx, gm_sigma_mes, alpha=0.8, color=my_colors, width=0.5, label=label)
    # bar3 = ax3.bar(indx, gm_sigma_rmse, color=my_colors, width=0.5, label=label, alpha=0.9)
    # ax3.set_title(r"$||\hat{\Sigma}_{GM(S, \lambda)} - \hat{\Sigma}_{GM(\hat{\Sigma}, \lambda)}||_F / p$",
    #               fontweight="bold")
    # ax3.set_xticks(indx)
    # ax3.set_xticklabels(auxi_labels, rotation=40)
    # ax3.set_ylabel("Frobenius norm")
    # ax3.set_ylim([0, max(gm_sigma_rmse) * 1.2])
    # autolabel(bar3, [round(x, 3) for x in gm_sigma_rmse], ax3)
    #
    # bar4 = ax4.bar(indx, gm_sigma_corr, color=my_colors, width=0.5, label=label, alpha=0.9)
    # ax4.set_title("Correlation of $\hat{\Sigma}_{GM(S, \lambda)} - \hat{\Sigma}_{GM(\hat{\Sigma}, \lambda)}$",
    #               fontweight="bold")
    # ax4.set_xticks(indx)
    # ax4.set_xticklabels(auxi_labels, rotation=40)
    # ax4.set_ylabel("Matrix correlation")
    # ax4.set_ylim([0, max(gm_sigma_corr) * 1.2])
    # autolabel(bar4, [round(x, 3) for x in gm_sigma_corr], ax4)
    #
    # bar5 = ax5.bar(indx, gm_prec_rmse, color=my_colors, width=0.5, label=label)
    # ax5.set_title(r"$||\hat{\Theta}_{S, \lambda} - \hat{\Theta}_{\lambda}||_F / p$", fontweight="bold")
    # ax5.set_xticks(indx)
    # ax5.set_xticklabels(auxi_labels, rotation=40)
    # ax5.set_ylabel("Frobenius norm")
    # ax5.set_ylim([0, max(gm_prec_rmse) * 1.2])
    # autolabel(bar5, [round(x, 3) for x in gm_prec_rmse], ax5)
    #
    # bar6 = ax6.bar(indx, gm_prec_corr, color=my_colors, width=0.5, label=label)
    # ax6.set_title(r"Correlation of $\hat{\Theta}_{S, \lambda}, \hat{\Theta}_{\lambda}$", fontweight="bold")
    # ax6.set_xticks(indx)
    # ax6.set_xticklabels(auxi_labels, rotation=40)
    # ax6.set_ylabel("Matrix correlation")
    # ax6.set_ylim([0, max(gm_prec_corr) * 1.2])
    # autolabel(bar6, [round(x, 3) for x in gm_prec_corr], ax6)

    plt.subplots_adjust(hspace=0.7)
    plt.show()

  avg_auxi_rmse = np.mean(avg_auxi_rmse, axis=0)
  avg_auxi_corr = np.mean(avg_auxi_corr, axis=0)

  fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(15, 15))
  bar1 = ax1.bar(indx, avg_auxi_rmse, color=my_colors, width=0.5)
  ax1.set_title(r"Sample Covariance RMSE", fontweight="bold", fontsize=18, y=1.005)
  ax1.set_xticks(indx)
  ax1.set_xticklabels(auxi_labels, rotation=40, fontsize=12)
  ax1.set_ylabel("RMSE", fontsize=13)
  ax1.set_ylim([0, max(avg_auxi_rmse) * 1.2])
  autolabel(bar1, [round(x, 3) for x in avg_auxi_rmse], ax1)

  bar2 = ax2.bar(indx, avg_auxi_corr, color=my_colors, width=0.5)
  ax2.set_title("Off-diagonal Correlation of $S, \hat{S}$", fontweight="bold", fontsize=18, y=1.005)
  ax2.set_xticks(indx)
  ax2.set_xticklabels(auxi_labels, rotation=40, fontsize=12)
  ax2.set_ylabel("Matrix correlation", fontsize=13)
  ax2.set_ylim([0, max(avg_auxi_corr) * 1.2])
  autolabel(bar2, [round(x, 3) for x in avg_auxi_corr], ax2)
  plt.subplots_adjust(hspace=0.6)

  # Plot accuracy, precision, recall and F1
  avg_acc = np.mean(avg_acc, axis=0)
  avg_prec = np.mean(avg_prec, axis=0)
  avg_recall = np.mean(avg_recall, axis=0)
  avg_f1 = np.mean(avg_f1, axis=0)
  plot_adj_evals_diff_auxi(auxi_labels, avg_acc, avg_prec, avg_recall, avg_f1)

    # # Plot adjacency matrices with different combo of auxiliary variables
    # for i in range(len(adj_mats)):
    #   plot_adj_mat(A=adj_mats[i], label=label + "" + auxi_labels[i], include_negs=False)


def autolabel(bar_plot, bar_label, ax):
  for idx, rect in enumerate(bar_plot):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
            bar_label[idx],
            ha='center', va='bottom', rotation=0, fontsize=12)


def plot_adj_evals_diff_auxi(auxi_labels, acc, prec, recall, f1):
  fig, ((ax1), (ax2), (ax3), (ax4)) = plt.subplots(4, 1, figsize=(15, 28))
  indx = np.arange(len(auxi_labels))
  if len(auxi_labels) == 6:
    my_colors = ["C7", "C0", "C9", "C1", "C8", "C3"]
  elif len(auxi_labels) == 11:
    my_colors = ["C7", "C0", "C9", "C1", "C8", "C0", "C9", "C1", "C8",
                 "C4", "C3"]
  else: #lif len(auxi_labels) == 16:
    my_colors = ["C7",
                 "C0", "C9", "C1", "C8", "C0", "C9", "C1", "C8", "C0", "C9", "C1", "C8",
                 "C6", "C4", "C3"]

  bar1 = ax1.bar(indx, acc, alpha=0.75, color=my_colors, width=0.5)
  ax1.set_title("Accuracy of Recovered Adjacency Matrix", fontweight="bold", fontsize=18, y=1.005)
  ax1.set_xticks(indx)
  ax1.set_xticklabels(auxi_labels, rotation=50)
  ax1.set_ylabel("Accuracy", fontsize=13)
  ax1.set_ylim([0, 1.1])
  autolabel(bar1, [round(x, 3) for x in acc], ax1)

  bar2 = ax2.bar(indx, prec, alpha=0.75, color=my_colors, width=0.5)
  ax2.set_title("Precision of Recovered Adjacency Matrix", fontweight="bold", fontsize=18, y=1.005)
  ax2.set_xticks(indx)
  ax2.set_xticklabels(auxi_labels, rotation=50)
  ax2.set_ylabel("Precision", fontsize=13)
  ax2.set_ylim([0, max(prec) * 1.1])
  autolabel(bar2, [round(x, 3) for x in prec], ax2)


  bar3 = ax3.bar(indx, recall, alpha=0.75, color=my_colors, width=0.5)
  ax3.set_title("Recall of Recovered Adjacency Matrix", fontweight="bold", fontsize=18, y=1.005)
  ax3.set_xticks(indx)
  ax3.set_xticklabels(auxi_labels, rotation=50)
  ax3.set_ylabel("Recall", fontsize=13)
  ax3.set_ylim([0, max(recall) * 1.1])
  autolabel(bar3, [round(x, 3) for x in recall], ax3)

  bar4 = ax4.bar(indx, f1, alpha=0.75, color=my_colors, width=0.5)
  ax4.set_title("F1 Score of Recovered Adjacency Matrix", fontweight="bold", fontsize=18, y=1.005)
  ax4.set_xticks(indx)
  ax4.set_xticklabels(auxi_labels, rotation=50, fontsize=12)
  ax4.set_ylabel("F1", fontsize=13)
  ax4.set_ylim([0, max(f1) * 1.2])
  autolabel(bar4, [round(x, 3) for x in f1], ax4)

  plt.subplots_adjust(hspace=0.6)
  plt.show()



# Plot 3 precision matrices
# def plot_3_precs(p0, p1, p2, r0, r1, r2, ax=None, standardize_prec=True):
#
#   plot_prec(p0, r0, standardize=standardize_prec,
#             ax=ax, label="Full Sample")
#
#   plot_prec(p1, r1, standardize=standardize_prec,
#             ax=ax, label="RLA Uniform")
#
#   plot_prec(p2, r2, standardize=standardize_prec,
#             ax=ax, label="RLA Minvar")
#
#
# def plot_3_adj_mats(A0, A1, A2, ax=None, include_negs=False):
#
#   plot_adj_mat(A0, ax, label="Full Sample", include_negs=include_negs)
#   plot_adj_mat(A1, ax, label="RLA Uniform", include_negs=include_negs)
#   plot_adj_mat(A2, ax, label="RLA Minvar", include_negs=include_negs)
#
#
# def plot_metric_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
#   plt.figure(figsize=(12, 9))
#   metric_uni = [[] for _ in cs]
#   metric_minvar = [[] for _ in cs]
#   for sidx in session_ids:
#     for ci in range(len(cs)):
#       c = cs[ci]
#       metric_uni[ci].extend(metric_all[sidx]["uni"][c])
#       metric_minvar[ci].extend(metric_all[sidx]["minvar"][c])
#   metric_uni = [np.mean(f) for f in metric_uni]
#   metric_minvar = [np.mean(f) for f in metric_minvar]
#
#   plt.plot(cs, metric_uni, "o-", label="RLA uniform")
#   plt.plot(cs, metric_minvar, "o-", label="RLA minvar")
#   #plt.title("%s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
#   plt.xlabel("RLA Sampling Percentage", fontsize=18)
#   plt.ylabel(metric_name, fontsize=18)
#   plt.legend(prop={'size': 20})
#
#   if save_name:
#     plt.savefig(save_name)
#   plt.show()
#
#
# def plot_metric_uni_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
#   # assert cs is sorted ascendingly
#   plt.figure(figsize=(12, 9))
#   for sidx in session_ids:
#     metric = metric_all[sidx]
#     mean_metric_uni = [np.mean(metric["uni"][c]) for c in cs]
#     plt.plot(cs, mean_metric_uni, "o-", label="RLA uniform [Session %d]" % sidx)
#
#   #plt.title("Uniform RLA Sampling %s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
#   plt.xlabel("RLA Sampling Percentage", fontsize=18)
#   plt.ylabel(metric_name, fontsize=18)
#   plt.legend(prop={'size': 13})
#
#   if save_name:
#     plt.savefig(save_name)
#   plt.show()
#
#
# def plot_metric_minvar_over_cs_all_sessions(cs, metric_all, metric_name, session_ids, neuron_cnts=None, save_name=None):
#   # assert cs is sorted ascendingly
#   plt.figure(figsize=(12, 9))
#   for sidx in session_ids:
#     metric = metric_all[sidx]
#     mean_frobs_minvar = [np.mean(metric["minvar"][c]) for c in cs]
#     plt.plot(cs, mean_frobs_minvar, "o-", label="RLA min-var [Session %d]" % sidx)
#
#   #plt.title("Min-var RLA Sampling %s [%d neurons]" % (metric_name, neuron_cnts), fontsize=20)
#   plt.xlabel("RLA Sampling Percentage", fontsize=18)
#   plt.ylabel(metric_name, fontsize=18)
#   plt.legend(prop={'size': 13})
#
#   if save_name:
#     plt.savefig(save_name)
#   plt.show()
#
#
# # Plot matrix correlation over different c
# def plot_corrs_over_cs(cs, corrs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, corrs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, corrs["minvar"], "o-", label="RLA minvar")
#   plt.title("Precision Matrix Correlation [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"Matrix Correlation", fontsize=18)
#   plt.legend(prop={'size': 13})
#
#   plt.show()
#
#
# # Plot edge estimation accuracy over different c
# def plot_acc_over_cs(cs, accs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, accs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, accs["minvar"], "o-", label="RLA minvar")
#   plt.title("Graph Structure Accuracy [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"Graph Structure Accuracy", fontsize=18)
#   plt.legend(prop={'size': 13})
#   plt.show()
#
#
# # Plot TPR over different c
# def plot_tpr_over_cs(cs, tprs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, tprs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, tprs["minvar"], "o-", label="RLA minvar")
#   plt.title("True Positive Rate of Edge Estimation [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"TPR", fontsize=18)
#   plt.legend(prop={'size': 13})
#   plt.show()
#
#
# # Plot FDR over different c
# def plot_fdr_over_cs(cs, fdrs, neuron_cnts):
#   plt.figure(figsize=(12, 9))
#   plt.plot(cs, fdrs["uni"], "o-", label="RLA uniform")
#   plt.plot(cs, fdrs["minvar"], "o-", label="RLA minvar")
#   plt.title("False Discovery Rate of Edge Estimation [n = %d neurons]" % neuron_cnts, fontsize=20)
#   plt.xlabel("RLA Sample Size", fontsize=18)
#   plt.ylabel(r"FDR", fontsize=18)
#   plt.legend(prop={'size': 13})
#   plt.show()
#

