
# Deep Learning-based Detection and Identification of Brain Tumor Biomarkers in Quantitative MR-Images

This repository contains code to support the study of deep learning techniques for detecting and segmenting brain tumor biomarkers using quantitative MR images (qMRI). The project investigates whether qMRI provides additional information compared to conventional MRI (cMRI) when detecting and segmenting brain tumors.

[Journal Article](https://dx.doi.org/10.1088/2632-2153/acf095) | [Cite](#reference)


**Abstract**

The infiltrative nature of malignant gliomas leads to active tumor spread into the peritumoral edema, which is undetectable with conventional MRI (cMRI) even after contrast injection. Quantitative MR relaxometry (qMRI) measures tissue-specific relaxation rates and may provide additional contrast mechanisms to detect non-enhancing infiltrative tumor areas. This study investigates whether qMRI data offers supplementary information compared to cMRI when used in deep learning-based brain tumor detection and segmentation models.
For this purpose, both preoperative conventional (T1-weighted pre- and post-contrast, T2-weighted, and FLAIR) and quantitative (pre- and post-contrast R1, R2, and proton density) MR images were collected from 23 patients showing radiological features consistent with high-grade glioma. The dataset includes 528 transversal slices used to train 2D deep learning models for tumor detection and segmentation with either cMRI or qMRI. Additionally, trends in the quantitative R1 and R2 rates of regions deemed important for tumor detection by model explainability methods were analyzed qualitatively.
Results show that models trained using a combination of pre- and post-contrast qMRI data achieved the highest performance metrics in tumor detection (MCC = 0.72) and segmentation (DSC = 0.90), though the difference compared to cMRI-based models was not statistically significant. Analysis of regions identified by model explainability techniques did not reveal notable differences between cMRI and qMRI-trained models. However, for individual cases, relaxation rates in areas outside of the annotated tumor regions exhibited similar post-contrast changes as those within the annotation, suggesting the presence of infiltrative tumor beyond cMRI-visible boundaries.
In conclusion, models trained with qMRI data achieved comparable detection and segmentation performance to those trained with cMRI, with the added advantage of quantitatively assessing brain tissue properties within similar scan times. For individual patients, the relaxation rate analysis in model-identified areas indicates potential infiltrative tumor regions beyond the cMRI-based annotations.

**Key highlights:**
- **Tumor detection and segmentation** performance were similar across models using conventional and quantitative MR data post-contrast. 
- **Across the cohort**, relevant regions for tumor detection did not show similarities in the relaxation rate trends within and outside the tumor annotation.
- **Single-subject analysis** reveals cases where relevant regions inside and outside the annotation have similar relaxation trends.


## Table of Contents
- [Setup](#Setup)
- [Datasets](#datasets)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [Reference](#reference)
- [License](#license)
---
## Setup
TODO. Add also FSL setup.

## Datasets
The clinical data used in this study cannot be made publicly available given (1) that it contains sensitive personal information and (2) the legal restrictions related to ethical approval. Nevertheless, one can experiment with the model training, evaluation, and explainability analysis using datasets such as [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html).

## Code structure
The code is organized in three folders:
- **Data_preprocessing_scripts**: contains the scripts needed for dataset preprocessing and organization for deep learning model training.
- **Tumor_detection_scripts**: Contains the scripts for model training and evaluation when looking at the brain tumor detection task. It also contains the scripts for plotting the results and Grad-CAM. 
- **Tumor_segmentation_scripts**: Contains the scripts for model training and evaluation when looking at the brain tumor detection task. It also contains the scripts for plotting the results.

## Usage
### Dataset preprocessing
**Registration to T1w-GD and skull stripping**: use the ``register_volumes_parallel_processing.sh`` script to register all the MR sequences available for each subject to the T1w-Gd volume. This script uses the FLIRT utility in the FSL library. After registration, a brain mask is obtained from the T1w-Gd volume and applied to all other registered MR sequences. Brain masking is performed using the BET utility in the FSL library. 

### Tumor detection
To configure and run tumor detection model training use the ``training_routine.py`` script available in the Tumor_detection/training_scripts folder. The script receives several in-line inputs such as the IMG_DATASET_FOLDER specifying the location of the dataset for training and DATASET_CONFIGURATION specifying the combination of MR sequences to use for training. For the full list of available settings run:
```bash
python3 Tumor_detection_scripts/training_scripts/training_routine.py --help
```
For an example of how to train the detection model through a repeated cross-validation scheme, see the ``bash_utility_run_model_training_for_ten_times_repeated_cross_validation.sh`` script.

After model training, model performance as tabular .csv data can be obtained using the ``gather_tabular_data.py`` script available in the Tumor_detection/plotting_scripts folder. The script can gather the test results from several trained model configurations (i.e. MR sequence combinations). 
The resulting tabular information can then be used to plot graphs for several metrics using the ``plot_boxplot_comparison_from_summary_file.py`` or ``plot_ROC_comparison_from_summary_file.py`` scripts. 
Additionally, scripts for obtaining occlusion and Grad-CAM attribution maps can be obtained using the ``plot_model_occlusionMaps.py`` and ``plot_model_GradCAM.py`` scripts. 
For a detailed description of how to run the different scripts, see the comments within each script.

### Tumor segmentation
To configure and run tumor detection model training use the ``training_routine.py`` script available in the Tumor_segmentation_scripts/training_scripts folder. The script receives several in-line inputs such as the IMG_DATASET_FOLDER specifying the dataset's location for training and DATASET_CONFIGURATION specifying the combination of MR sequences to use for training. For the full list of available settings run:
```bash
python3 Tumor_segmentation_scripts/training_scripts/training_routine.py --help
```
For an example of how to train the detection model through a repeated cross-validation scheme, see the ``run_five_times_repeated_cross_validation.sh`` script.

After training, models can be tested using the ``test_model.py`` script. Then, using the ``gather_tabular_data.py`` scipt, the testig results from several model configurations (i.e. MR sequence input combinations) can be aggregated in a tabular .csv file. 
Run the ``plot_boxplot_comparison_from_summary_file.py`` script on the tabular file to plot the comparison between the different MR sequence configurations.
For a detailed description of how to run the different scripts, see the comments within each script.

## Reference
If you use this work, please cite:

```bibtex
@article{tampu_2023_biomarker,
doi = {10.1088/2632-2153/acf095},
url = {https://dx.doi.org/10.1088/2632-2153/acf095},
year = {2023},
month = {sep},
publisher = {IOP Publishing},
volume = {4},
number = {3},
pages = {035038},
author = {Iulian Emil Tampu and Neda Haj-Hosseini and Ida Blystad and Anders Eklund},
title = {Deep learning-based detection and identification of brain tumor biomarkers in quantitative MR-images},
journal = {Machine Learning: Science and Technology},}
```

## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
