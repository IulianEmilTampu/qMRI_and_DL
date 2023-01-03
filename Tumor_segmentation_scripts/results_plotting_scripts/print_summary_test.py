'''
Script that reads the .csv files created using the get_overall_model_test.py script
and prints a summary for each model
'''

import os
import csv
import numpy as np

## get path to the csv file
csv_path = "/home/iulta54/Code/P4-qMRI/"
csv_files = [
            "model-BRATS_summary_model_test.csv",
            "model-qMRI_summary_model_test.csv",
            "model-qMRI_GD_summary_model_test.csv"
            ]

## take out data
summary_dict = {}

for file in csv_files:
    # find model name from file
    model_name = file[file.find("model-")+6:file.find("_summary")]
    # create item in dictionary
    summary_dict[model_name]= {
                                "train":{},
                                "validation":{},
                                "test":{},
                                }
    # open file and get all the data
    with open(os.path.join(csv_path,file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        aus_list = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                # save information in the right spot
                # group values per subject and per set
                if any([row[1]==key for key in summary_dict[model_name][row[0]].keys()]):
                    # subject already present, append values
                    summary_dict[model_name][row[0]][row[1]]["dice"].append(float(row[3]))
                    summary_dict[model_name][row[0]][row[1]]["hd95"].append(float(row[4]))
                    summary_dict[model_name][row[0]][row[1]]["tp"].append(float(row[5]))
                    summary_dict[model_name][row[0]][row[1]]["fp"].append(float(row[6]))
                    summary_dict[model_name][row[0]][row[1]]["tn"].append(float(row[7]))
                    summary_dict[model_name][row[0]][row[1]]["fn"].append(float(row[8]))
                else:
                    # subject not seen. Initialize
                    summary_dict[model_name][row[0]][row[1]] = {
                                            "dice":[float(row[3])],
                                            "hd95":[float(row[4])],
                                            "tp":[float(row[5])],
                                            "fp":[float(row[6])],
                                            "tn":[float(row[7])],
                                            "fn":[float(row[8])]
                                            }

## Print performances for each set as mean over the subjects

for model, values in summary_dict.items():
    for set, subjects in values.items():
        mean_dice = np.mean( [np.mean(s["dice"]) for s in subjects.values() ] )
        std_dice  = np.std( [np.mean(s["dice"]) for s in subjects.values() ] )

        mean_hd95 = np.mean( [np.mean(s["hd95"]) for s in subjects.values() ] )
        std_hd95  = np.std( [np.mean(s["hd95"]) for s in subjects.values() ] )

        mean_accuracy = np.mean( [(np.sum(s["tp"])+np.sum(s["tn"]))/(np.sum(s["tp"])+np.sum(s["tn"]+np.sum(s["fp"])+np.sum(s["fn"]))) for s in subjects.values() ] )
        std_accuracy  = np.std( [(np.sum(s["tp"])+np.sum(s["tn"]))/(np.sum(s["tp"])+np.sum(s["tn"]+np.sum(s["fp"])+np.sum(s["fn"]))) for s in subjects.values() ] )

        mean_precision = np.mean( [np.sum(s["tp"])/(np.sum(s["tp"])+np.sum(s["fp"])+1e-8) for s in subjects.values() ] )
        std_precision  = np.std( [np.sum(s["tp"])/(np.sum(s["tp"])+np.sum(s["fp"])+1e-8) for s in subjects.values() ] )

        mean_recall = np.mean( [np.sum(s["tp"])/(np.sum(s["tp"])+np.sum(s["fn"])) for s in subjects.values() ] )
        std_recall  = np.std( [np.sum(s["tp"])/(np.sum(s["tp"])+np.sum(s["fn"])) for s in subjects.values() ] )

        # print results for this model and set
        print(f'Model - {model:9s} - {set:8} set \n'
                f'{" "*32}{"Dice":12s}{"Hd95":12s}{"Accuracy":12s}{"Precision":12s}{"Recall":12s}\n'
                f'{" "*20}{"mean":12s}{mean_dice:<03.04f}{" "*5}{mean_hd95:<03.04f}{" "*5}{mean_accuracy:^03.04f}{" "*5}{mean_precision:03.04f}{" "*5}{mean_recall:03.04f}{" "*5}\n'
                f'{" "*20}{"std":12s}{std_dice:.04f}{" "*5}{std_hd95:.04f}{" "*5}{std_accuracy:.04f}{" "*5}{std_precision:.04f}{" "*8}{std_recall:.04f}{" "*5}\n'
                f'\n\n'
                )


