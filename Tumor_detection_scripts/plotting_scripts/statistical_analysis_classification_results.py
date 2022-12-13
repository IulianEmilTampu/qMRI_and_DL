# %%
"""
Script that uses the test results saved as tabular infromation from all the trained folds and input modalities (otained through the 
gather_tabular_data.py script) to perform statistical comparison between the performance of the models.

Steps
1 - load tabular data and parse information
2 - specify comparisons to be made
3 - perform statistical comparison
"""

import csv
from xml.parsers.expat import model
from scipy import stats
import pandas as pd
import numpy as np
import argparse

# %%
su_debug_flag = True

# --------------------------------------
# read the input arguments and set the base folder
# --------------------------------------
if not su_debug_flag:
    parser = argparse.ArgumentParser(
        description="Script that performs statistical comparison of model performance for models trained on different input configurations."
    )
    parser.add_argument(
        "-ptm",
        "--SUMMARY_FILE_PATH",
        required=True,
        help="Path to the overall_tabular_test_summary.csv file",
    )
    args_dict = dict(vars(parser.parse_args()))
else:
    print("Running in debug mode.")
    args_dict = {
        "SUMMARY_FILE_PATH": "/flush/iulta54/Research/P4-qMRI_git/Tumor_detection_scripts/trained_models_archive/overall_tabular_test_summary.csv",
    }

# print input variables
max_len = max([len(key) for key in args_dict])
[print(f"{key:{max_len}s}: {value}") for key, value in args_dict.items()]

# %% GET FILE AND PARSE
DATA = pd.read_csv(args_dict["SUMMARY_FILE_PATH"])

# %% SPECIFY COMPARISONS TO BE MADE AND SIGNIFICANCE TRESHOLD
import itertools

list_of_comparisons = list(
    itertools.combinations(["cMRI", "qMRI", "qMRIGD", "qMRI-qMRIGD"], 2)
)
# list_of_comparisons = [["BRATS", "qMRI"], ["BRATS", "qMRIGD"], ["qMRI", "qMRIGD"], []]
metrics_to_compare = ["matthews_correlation_coefficient", "accuracy", "auc"]
model_version = "last"

significance_thr = 0.05
bonferroni_corrected_significance_thr = significance_thr / len(list_of_comparisons)

# %% PERFORM STATISTICAL COMPARISON AND PRINT RESULTS
comparison_results = dict.fromkeys(
    range(len(list_of_comparisons)), dict.fromkeys(metrics_to_compare)
)
print(
    f"Significance threshold: {bonferroni_corrected_significance_thr} (Bonferroni correction for {len(list_of_comparisons)} comparisons)"
)
for idx, comparison in enumerate(list_of_comparisons):
    print(
        f"############ Comparison between {comparison[0]} and {comparison[1]} #################"
    )
    max_len = max([len(i) for i in comparison])
    for m in metrics_to_compare:
        print(f"Metric -> {m}")
        # get populations to be compared
        population_1 = list(
            DATA.loc[
                (DATA["input_type"] == comparison[0])
                & (DATA["model_version"] == model_version)
            ][m]
        )
        population_2 = list(
            DATA.loc[
                (DATA["input_type"] == comparison[1])
                & (DATA["model_version"] == model_version)
            ][m]
        )
        statistical_test = stats.wilcoxon(
            population_1, population_2, alternative="two-sided"
        )
        # # perform test adn save results
        # comparison_results[idx][m] = {
        #     "statistical_test": stats.ttest_ind(
        #         population_1, population_2, alternative="two-sided"
        #     ),
        #     "population_1_values": population_1,
        #     "population_2_values": population_2,
        # }

        # print resutls
        print(
            f"    {comparison[0]:{max_len}s} mean±std: {np.mean(population_1):0.4f} ± {np.std(population_1):0.4f}"
        )
        print(
            f"    {comparison[1]:{max_len}s} mean±std: {np.mean(population_2):0.4f} ± {np.std(population_2):0.4f}"
        )
        print(
            f'    p-value: {statistical_test[-1]:0.8f} ({"SIGNIFICANT" if statistical_test[-1] <= bonferroni_corrected_significance_thr else "NOT SIGNIFICANT"})'
        )
# %%
