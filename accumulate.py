import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_report_signals(sample_path):
    """Return the signals for a sample."""
    
    # Read lines from the report00.csv file
    with open(Path(sample_path, "Report00.CSV"), "r", encoding="utf-16") as file:
        lines = file.readlines()
    
    # Dict to track which report files have which wavelength
    report_signals = {}
    for line in lines:
        # Pattern that matches the signal line
        pat = r'"Signal (?P<num>\d)","DAD1 \w, Sig=(?P<wavelength>\d{3}),4 Ref=off",""'
        # Catch error when re.match finds no match and returns None
        try:
            # Attempt to get groups
            report_num, wavelength = re.match(pat, line).groups()
        except AttributeError:
            # No match was found, move on to next line
            continue

        # Add the report data to signals
        report_signals.update({f"REPORT0{report_num}.CSV": wavelength})
    return report_signals


def get_report_header(sample_path):
    """Return the header (names of columns) for a sample."""

    # Read lines from the report00.csv file
    with open(Path(sample_path, "Report00.CSV"), "r", encoding="utf-16") as file:
        lines = file.readlines()
    
    # Dict to store header
    header = []
    for line in lines:
        # Pattern that matches the column line
        pat = r'"Column \d","(?P<col_name>[\w\s\%]+)",".*"'
        # Catch error when re.match finds no match and returns None
        try:
            # Attempt to get groups
            col_name = re.match(pat, line).groups()[0].strip()
        except AttributeError:
            # No match was found, move on to next line
            continue

        # Add the column name to header
        header.append(col_name)
    return header


def get_chem_classes(experiment_df):
    """Return a list of predicted chemical classes."""
    # Create a df of features for the model to predict from
    predict_df = experiment_df[["Retention Time", "218nm Area", "250nm Area", "260nm Area", "330nm Area", "350nm Area"]]

    # Replace NaN values with the median for that column
    for column_name in predict_df.columns:
        if predict_df[column_name].dtype != "object":
            predict_df[column_name] = predict_df[column_name].replace(np.NaN, predict_df[column_name].median())

    # Predict each rows chemical class
    return logit_model.predict(predict_df)


def get_report_dfs(sample_path, report_names, report_header):
    report_dfs = []
    for report_name in report_names:
        report_df = pd.read_csv(Path(sample_path, report_name),
                                names=report_header,
                                encoding="utf-16")
        # Round retention time to 1 decimal place
        report_df["Retention Time"] = report_df["Retention Time"].round(1)
        report_dfs.append(report_df)
    return report_dfs


def get_invalid_rows(experiment_df):
    """Return a list of invalid indices in expeirment df. A row is
    invalid if it has a minimum peak area of less than 400 or contains
    more than one NaN value.
    """
    drop_is = []
    for i in experiment_df.index:
        # Get the row at this index
        row = experiment_df.loc[i]
        # Drop rows with low areas
        if row[wavelength_names].min() < 200:
            drop_is.append(i)
        # Drop rows with 2 or more NaNs
        elif row.isna().sum() > 1:
            drop_is.append(i)
        elif row["Retention Time"] < 3.0:
            drop_is.append(i)
    return drop_is


# Load the training data for our classifier model
training_df = pd.read_csv("train.csv")
X = training_df[["Retention Time", "218nm Area", "250nm Area", "260nm Area", "330nm Area", "350nm Area"]]
y = list(training_df["ID"])

# Create and train the model
logit_model = LogisticRegression().fit(X, y)

# List of experiments to be accumulated
experiments = [
	"Plate2_9WeekAndMonarchs_11_9_232023-11-0914-43-29",
	"New_GV2023-10-1915-57-44",
	"New2023-11-0609-22-32",
	"Herbiv_prepost_ALL_sumr232023-11-0216-42-05"
]
# Loop through all the experiments, acculumating their data
for experiment in experiments:
    # Get the paths to all samples contained within this experiment
    sample_paths = [Path(experiment, sample_name) for sample_name in os.listdir(experiment)
                    if sample_name[-2:] == ".D"]
    
    # Assume all samples within this experiment have the same signals,
    # which correspond to the same report files.
    report_signals = get_report_signals(sample_paths[0])

    # Assume all samples' report files have the same headers
    report_header = get_report_header(sample_paths[0])

    # Create the dataframe to store this experiment's data
    columns = ["Sample Name", "Retention Time"]
    wavelength_names = [wavelength + "nm Area" for wavelength in report_signals.values()]
    columns.extend(wavelength_names)
    experiment_df = pd.DataFrame(columns=columns)

    # Loop through all sample paths
    for sample_path in sample_paths:
        # Get the sample name from the path
        sample_name = sample_path.parts[-1]
        
        # Dictionaries are not ordered so sorting is necessary to
        # ensure constant iteration over the keys.
        report_names = sorted(report_signals.keys())
        # Load all report files as dataframes into this list
        report_dfs = get_report_dfs(sample_path, report_names, report_header)
        
        # Make a set of all (unique) retention times
        unique_ret_times = set()
        for report_df in report_dfs:
            unique_ret_times.update(report_df["Retention Time"])

        # Loop through retention times in the first report file
        for ret_time in unique_ret_times:
            # List of areas for this retention time accross
            # the different report files
            areas = []
            for report_df in report_dfs:
                # Get rows from this report dataframe that have
                # matching retention times
                matching_rows = report_df[report_df["Retention Time"] == ret_time]

                # If there is only one matching row, use the area from
                # that row
                if len(matching_rows) == 1:
                    areas.append(*matching_rows["Area"])
                # Otherwise, use NaN
                else:
                    areas.append(np.NaN)
            
            # Assemble and add the new row to the experiment dataframe
            new_row = [sample_name, ret_time] + areas
            experiment_df.loc[len(experiment_df.index)] = new_row

    # Drop rows that are invalid
    invalid_rows = get_invalid_rows(experiment_df)
    experiment_df.drop(labels=invalid_rows, axis=0, inplace=True)

    # Add the chemical class column
    experiment_df["Chemical Class"] = get_chem_classes(experiment_df)

    # Save the experiment dataframe
    experiment_df.to_csv(f"acc_{experiment}.csv", index=False)
    
    # Write a completion message to the console
    print(f"{experiment} accumulation completed")
