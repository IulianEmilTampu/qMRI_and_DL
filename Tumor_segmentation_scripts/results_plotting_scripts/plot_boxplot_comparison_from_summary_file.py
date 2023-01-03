# %%
"""
Script that uses the overall_tabular_test_summary.csv file to plot the results
as boxplots (other implementations can be added).
"""
import os
import glob
import csv
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle


# %% utilities
def add_median_labels(ax, precision=".3f"):
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


def add_ensemble_values(
    df_5folds, df_ensemble, ax, hue_order, metric_to_plot, one_marker=False
):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    # define markers to use
    if one_marker:
        available_markers = ["X"]
    else:
        available_markers = ["s", "v", "^", "p", "X", "8", "*"]
    markers = cycle([h for h in available_markers[0 : len(hue_order)]])

    # get list of unique classifications
    unique_classifications = df_5folds.classification_type.unique()
    # make some tricks to be able to index the df_ensamble later
    unique_classifications = [
        x for x in unique_classifications for _ in range(len(hue_order))
    ]
    models = cycle(hue_order)

    # loop through the different boxes
    for idx, median in enumerate(lines[4 : len(lines) : lines_per_box]):
        # get x location of the box
        x = median.get_data()[0].mean()
        # get y location based on the value of the ensemble
        # # get which model we are looking at
        m = next(models)
        y = float(
            df_ensemble.loc[
                (df_ensemble["classification_type"] == unique_classifications[idx])
                & (df_ensemble["model_type"] == m)
            ][metric_to_plot]
        )
        ax.scatter(
            x, y, marker=next(markers), color="k", edgecolors="white", s=150, zorder=5
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

# %% BOXPLOT OF BEST WITH ENSEMBLE OVERLAYED
save_images = True

# plot params
# plt.rcParams["font.family"] = "Times New Roman"
tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 20

# select model version
model_version = "last"
# metricts to plot
metrics = [
    "Dice",
]
# metrics names in the plot
metric_name_for_plot = [
    "Dice [0,1]",
]

for metric_to_plot in metrics:
    print(metric_to_plot)
    # filter data to plot (model_version and NOT ensemble)
    df = DF.loc[(DF["model_version"] == model_version) & (DF["fold"] != "ensemble")]

    # create figure
    fig, box_plot = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # add boxplot for the filtered data
    box_plot = sns.boxplot(
        x="input_configuration",
        y=metric_to_plot,
        data=df,
        palette="Set3",
        showfliers=False,
    )
    # fix figure labels
    box_plot.set_title(f"Summary classification", fontsize=title_font_size)
    box_plot.tick_params(labelsize=tick_font_size)
    box_plot.set_ylabel(
        f"{metric_name_for_plot[metrics.index(metric_to_plot)].capitalize()}",
        fontsize=label_font_size,
    )
    # format y tick labels
    ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels)

    box_plot.set_xlabel("")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=0)
    # plt.setp(box_plot.get_legend().get_texts(), fontsize=legend_font_size)

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
        [h for h in available_hatches[0 : len(df.input_configuration.unique())]]
    )
    colors, legend_hatch = [], []

    # add pattern to boxplots and legend
    # # add hatches to box plots
    for (hatch, patch) in zip(hatches, box_plot.artists):
        # Boxes from left to right
        patch.set_hatch(hatch)
        patch.set_zorder(2)
    # add median
    add_median_labels(box_plot)
    box_plot.yaxis.grid(True, zorder=-3)  # Hide the horizontal gridlines
    box_plot.xaxis.grid(False)  # Show the vertical gridlines
    box_plot.figure.tight_layout()

    # add shadows
    y_min = box_plot.get_ylim()[0]
    y_max = box_plot.get_ylim()[-1]
    add_shadow_between_hues(box_plot, y_min, y_max, alpha=0.01, zorder=60, color="blue")

    if save_images == True:
        file_name = (
            f"overall_classification_summary_{model_version}_model_{metric_to_plot}"
        )
        fig.savefig(
            os.path.join(args_dict["SAVE_PATH"], file_name + ".pdf"),
            bbox_inches="tight",
            dpi=300,
        )
        fig.savefig(
            os.path.join(args_dict["SAVE_PATH"], file_name + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
    else:
        plt.show()

# %% PLOT ENSEMBLE

save_images = True

# plot params
# plt.rcParams["font.family"] = "Times New Roman"
tick_font_size = 17
title_font_size = 20
label_font_size = 20
legend_font_size = 20

# select model version
model_version = "last"
# metricts to plot
metrics = [
    "Dice",
]
# metrics names in the plot
metric_name_for_plot = [
    "Dice [0,1]",
]

for metric_to_plot in metrics:
    print(metric_to_plot)
    # filter data to plot (model_version and NOT ensemble)
    df = DF.loc[(DF["model_version"] == model_version) & (DF["fold"] == "ensemble")]

    # create figure
    fig, box_plot = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
    # add boxplot for the filtered data
    box_plot = sns.boxplot(
        x="input_configuration",
        y=metric_to_plot,
        data=df,
        palette="Set3",
        showfliers=False,
    )
    # fix figure labels
    box_plot.set_title(f"Summary classification", fontsize=title_font_size)
    box_plot.tick_params(labelsize=tick_font_size)
    box_plot.set_ylabel(
        f"{metric_name_for_plot[metrics.index(metric_to_plot)].capitalize()}",
        fontsize=label_font_size,
    )
    # format y tick labels
    ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels)

    box_plot.set_xlabel("")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=0)
    # plt.setp(box_plot.get_legend().get_texts(), fontsize=legend_font_size)

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
        [h for h in available_hatches[0 : len(df.input_configuration.unique())]]
    )
    colors, legend_hatch = [], []

    # add pattern to boxplots and legend
    # # add hatches to box plots
    for (hatch, patch) in zip(hatches, box_plot.artists):
        # Boxes from left to right
        patch.set_hatch(hatch)
        patch.set_zorder(2)
    # add median
    add_median_labels(box_plot)
    box_plot.yaxis.grid(True, zorder=-3)  # Hide the horizontal gridlines
    box_plot.xaxis.grid(False)  # Show the vertical gridlines
    box_plot.figure.tight_layout()

    # add shadows
    y_min = box_plot.get_ylim()[0]
    y_max = box_plot.get_ylim()[-1]
    add_shadow_between_hues(box_plot, y_min, y_max, alpha=0.01, zorder=60, color="blue")

    if save_images == True:
        file_name = f"overall_classification_summary_ensemble_{model_version}_model_{metric_to_plot}"
        fig.savefig(
            os.path.join(args_dict["SAVE_PATH"], file_name + ".pdf"),
            bbox_inches="tight",
            dpi=300,
        )
        fig.savefig(
            os.path.join(args_dict["SAVE_PATH"], file_name + ".png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)
    else:
        plt.show()
