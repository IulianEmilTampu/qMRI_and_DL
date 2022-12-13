# %%
"""
Script that given the .json summary file containing the logits for each input
configuration, plots the ROC comparison. 
The data in teh summary file should be structure like:
he data is strucrute as follows
INPUT_CONFIGURATION
    MODEL_VESRION : 'best', 'last'
        FOLD_NBR 
            test_prediction : list
            test_softmax : list

This can be obtained by testing the models using the test_models_v2.py script

Steps
1 - get summary file
2 - load data 
3 - plot
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import auc, roc_curve, roc_auc_score

# %% GET PATHS AND LOAD SUMMARY FILE
SUMMARY_FILE_PATH = "/flush/iulta54/Research/P4-qMRI/Manuscript_results/Tumor_detection/gathered_test_logits.json"
SAVE_PATH = (
    "/flush/iulta54/Research/P4-qMRI/Manuscript_results/Tumor_detection/Summary_plots"
)

with open(SUMMARY_FILE_PATH, "r") as file:
    summary_logits = json.load(file)

INPUT_CONFIGURATIONS = list(summary_logits.keys())

# %% GATHER TRUE-POSITIVE-RATE (TPR) AND FALSE-POSITIVE-RATE (FTR) FOR EACH INPUT CONFIGURATION
gathered_info = {}

for ic in INPUT_CONFIGURATIONS:
    gathered_info[ic] = {}
    for model_version in ["best", "last"]:
        gathered_info[ic][model_version] = {"tpr": [], "fpr": [], "auc": []}
        for f in range(len(summary_logits[ic][model_version])):
            y = np.array(
                summary_logits[ic][model_version][str(f + 1)]["per_sample_ground_truth"]
            )
            y_ = np.array(
                summary_logits[ic][model_version][str(f + 1)]["per_sample_logits"]
            )
            # compute tpr and fpr
            fpr, tpr, thresholds = roc_curve(y[:, 1], y_[:, 1])
            auc = roc_auc_score(y, y_, average=None)
            # save
            gathered_info[ic][model_version]["tpr"].append(tpr)
            gathered_info[ic][model_version]["fpr"].append(fpr)
            gathered_info[ic][model_version]["auc"].append(auc)

# bring every fold to have the same number of tpr and fpr
seq_lentgh = 100
for ic in INPUT_CONFIGURATIONS:
    for model_version in ["best", "last"]:
        for f in range(len(gathered_info[ic][model_version]["auc"])):
            fpr_new = np.linspace(0, 1, seq_lentgh)
            tpr_new = np.interp(
                fpr_new,
                gathered_info[ic][model_version]["fpr"][f],
                gathered_info[ic][model_version]["tpr"][f],
            )
            tpr_new[0] = 0
            tpr_new[-1] = 1
            # save
            gathered_info[ic][model_version]["tpr"][f] = tpr_new
            gathered_info[ic][model_version]["fpr"][f] = fpr_new

# %% PLOT MEAN ROC FOR EACH INPUT CONFIGURATION

save = True
model_version = "best"

# font settings
tick_font_size = 30
title_font_size = 20
label_font_size = 30
csfont = {"fontname": "Times New Roman"}
plt.rcParams["font.family"] = "Times New Roman"
# legend_font_size="xx-large"
legend_font_size = "xx-large"
line_width = 2

# set if plotting ranges
plot_ranges = False

list_colors = [
    "blue",
    "orange",
    "green",
    "gray",
    "purple",
    "teal",
    "pink",
    "brown",
    "red",
    "cyan",
    "olive",
]
list_styles = [
    "-",
    "--",
    "-.",
    ":",
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 10, 1, 10)),
    (0, (3, 10, 1, 10, 1, 10)),
]
colors = cycle(list_colors)
line_styles = cycle(list_styles)

mll = np.max([len(l) for l in INPUT_CONFIGURATIONS])

fig, ax = plt.subplots(figsize=(10, 10))
for color, line_stl, ic in zip(colors, line_styles, INPUT_CONFIGURATIONS):
    mean_tpr = np.mean(gathered_info[ic][model_version]["tpr"], axis=0)
    mean_fpr = np.mean(gathered_info[ic][model_version]["fpr"], axis=0)
    mean_auc = np.mean(gathered_info[ic][model_version]["auc"])
    std_auc = np.std(gathered_info[ic][model_version]["auc"])

    ax.plot(
        mean_fpr,
        mean_tpr,
        label=f"Mean ROC: {ic:{mll+1}s}(AUC:{mean_auc:0.3f} \u00B1 {std_auc:0.3f})",
        color=color,
        linestyle=line_stl,
        alpha=1,
        lw=line_width,
    )
    # plot ranges if needed
    if plot_ranges:
        lower_tpr = mean_tpr - np.std(gathered_info[ic][model_version]["tpr"], axis=0)
        upper_tpr = mean_tpr + np.std(gathered_info[ic][model_version]["tpr"], axis=0)
        ax.fill_between(
            mean_fpr,
            lower_tpr,
            upper_tpr,
            color=color,
            alpha=0.05,
        )

ax.tick_params(labelsize=tick_font_size)
plt.grid(color="b", linestyle="-.", linewidth=0.1, which="both")

ax.set_xlabel("False Positive Rate", fontsize=label_font_size)
ax.set_ylabel("True Positive Rate", fontsize=label_font_size)
ax.set_title("Comparison multi-class ROC", fontsize=title_font_size, **csfont)
ax.legend(loc="lower right", prop={"family": "monospace"})
plt.setp(ax.get_legend().get_texts(), fontsize=legend_font_size)

if save is True:
    fig.savefig(
        os.path.join(SAVE_PATH, f"comparison_ROCs_{model_version}_models.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    fig.savefig(
        os.path.join(SAVE_PATH, f"comparison_ROCs_{model_version}_models.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
else:
    plt.show()
