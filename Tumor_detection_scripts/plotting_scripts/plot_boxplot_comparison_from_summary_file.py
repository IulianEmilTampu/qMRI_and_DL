# %%
"""
Script that uses the overall_tabular_test_summary.csv file to plot the results
as boxplots (other implementations can be added).
"""
import os
import glob
import csv
import json
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle


# %% UTILITIES


def add_median_labels(ax, precision=".1f"):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4 : len(lines) : lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(
            x,
            y,
            f"{value:{precision}}",
            ha="center",
            va="center",
            fontweight="bold",
            color="black",
        )
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #     path_effects.Stroke(linewidth=2, foreground=median.get_color()),
        #     path_effects.Normal(),
        # ])
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=2, foreground="white"),
                path_effects.Normal(),
            ]
        )


def add_shadow_between_hues(ax, y_min, y_max, alpha=0.05, zorder=30, color="black"):
    # get hue region positions
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))

    # get the number of boxes per hue region
    nbr_boxes_per_hue = int(len(boxes) / len(ax.get_xticks()))

    # build coordinate regions where to add shadow
    # starting from the 0th or 1st hue gerion
    start_hue_region = 0
    # take the initial coordinate of the first box of the region and
    # the last of the last box of the region
    # get coordinatex for all boxes in order
    x_boxes_coordinates = []
    for idx, median in enumerate(lines[4 : len(lines) : lines_per_box]):
        # get x location of the box
        x_boxes_coordinates.append(median.get_data()[0])

    # get hue region coordinate
    hue_region_coordinatex = []
    for hue_region in range(len(ax.get_xticks())):
        idx_first_box = hue_region * nbr_boxes_per_hue
        idx_last_box = idx_first_box + nbr_boxes_per_hue - 1
        hue_region_coordinatex.append(
            [
                x_boxes_coordinates[idx_first_box][0],
                x_boxes_coordinates[idx_last_box][-1],
            ]
        )

    # loop through the regions and color
    for c in range(start_hue_region, len(ax.get_xticks()), 2):
        x_min, x_max = hue_region_coordinatex[c][0], hue_region_coordinatex[c][-1]
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                (x_max - x_min),
                (y_max - y_min),
                color=color,
                alpha=alpha,
                zorder=zorder,
            )
        )


# %%
# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------

su_debug_flag = True

if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Script that plots the comparison between the different imput modalities using the overall summary files obtained using the gater_tabular_data'.py scripts."
    )
    parser.add_argument(
        "-ptm",
        "--SUMMARY_FILE_PATH",
        required=True,
        help="Path to the overall_tabular_test_summary.csv file",
    )
    parser.add_argument(
        "-sp",
        "--SAVE_PATH",
        required=True,
        help="Provide the path where to save the summary .json file of the gathered information.",
    )

    args_dict = dict(vars(parser.parse_args()))
else:
    print("Running in debug mode.")
    args_dict = {
        "SUMMARY_FILE_PATH": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/trained_models_archive/overall_tabular_test_summary.csv",
        "SAVE_PATH": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/trained_models_archive/temp_summary_figures",
    }

Path(args_dict["SAVE_PATH"]).mkdir(parents=True, exist_ok=True)

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]


# load csv as dataframe to work easily with seaborn
DF = pd.read_csv(args_dict["SUMMARY_FILE_PATH"])

# %% BOXPLOT OF BEST MODELS WITH ENSEMBLE OVERLAYED

save_images = True
model_version = "best"
add_median_value = False

# font settings
tick_font_size = 20
title_font_size = 20
label_font_size = 30
csfont = {"fontname": "Times New Roman"}
plt.rcParams["font.family"] = "Times New Roman"
legend_font_size = "xx-large"

metrics = [
    "precision",
    "recall",
    "accuracy",
    "f1-score",
    "auc",
    "matthews_correlation_coefficient",
]
metric_name_for_plot = [
    "Precision [0,1]",
    "Recall [0,1]",
    "Accuracy [0,1]",
    "F1-score [0,1]",
    "AUC [0,1]",
    "Matthews correlation coefficient [-1,1]",
]

# ###################### debug
# metrics = ["matthews_correlation_coefficient"]
# metric_name_for_plot = [
#     "Matthews correlation coefficient [-1,1]",
# ]

for metric_to_plot in metrics:
    # get data by filtering the dataframe
    df_5folds = DF.loc[(DF["model_version"] == model_version)]
    unique_classificatios = df_5folds.classification_type.unique()

    # create figure and populate
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 8))
    box_plot = sns.boxplot(
        x="input_type",
        y=metric_to_plot,
        data=df_5folds,
        palette="Set3",
        showfliers=False,
        orient="v",
        ax=ax,
    )

    box_plot.set_title(f"Summary classification", fontsize=title_font_size)
    box_plot.tick_params(labelsize=tick_font_size)

    box_plot.set_ylabel(
        f"{metric_name_for_plot[metrics.index(metric_to_plot)].capitalize()}",
        fontsize=label_font_size,
    )
    # format y tick labels
    ylabels = [f"{x:,.1f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels)

    box_plot.set_xlabel("")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=0)

    # add pattern to boxplots
    available_hatches = [
        "//",
        "xx",
        "|",
        "oo",
        "-",
        "\\\\",
        ".",
        "/",
        "\\",
    ]
    hatches = cycle(
        [h for h in available_hatches[0 : len(df_5folds.input_type.unique())]]
    )
    colors, legend_hatch = [], []

    # add pattern to boxplots and legend
    # # add hatches to box plots
    for (hatch, patch) in zip(hatches, box_plot.artists):
        # Boxes from left to right
        patch.set_hatch(hatch)
        patch.set_zorder(2)

    if add_median_value:
        add_median_labels(box_plot, precision=".2f")

    # add shadows
    y_min = box_plot.get_ylim()[0]
    y_max = box_plot.get_ylim()[-1]
    add_shadow_between_hues(box_plot, y_min, y_max, alpha=0.01, zorder=60, color="blue")

    # final touches
    box_plot.yaxis.grid(True, zorder=0)  # Hide the horizontal gridlines
    box_plot.xaxis.grid(False)  # Show the vertical gridlines
    box_plot.figure.tight_layout()

    if save_images == True:
        file_name = (
            f"overall_classification_summary_{model_version}_model_{metric_to_plot}"
        )
        fig.savefig(
            os.path.join(args_dict["SAVE_PATH"], file_name + ".pdf"),
            bbox_inches="tight",
            dpi=100,
        )
        fig.savefig(
            os.path.join(args_dict["SAVE_PATH"], file_name + ".png"),
            bbox_inches="tight",
            dpi=100,
        )
        plt.close(fig)
    else:
        plt.show()

# %%
