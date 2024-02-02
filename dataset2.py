import re
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def get_meta_data(meta_df, experiment_name, sample_name):
    """Return the meta data for a sample_name as a list with the
    repetition number tacked on to the end. List values are ordered
    corresponding to the columns in meta_df.
    """
    # Each sample name should follow this pattern
    # pat = r"\d{3}-P\d-\w\d+-(?P<plate_loc>\w\d+)_(?P<family_num>\d+)-(?P<rep>[A-Z]*\d*)"  # works for Herbiv_prepost_ALL_sumr232023-11-0216-42-05
    pat = r'\d{3}-P\d-(?P<plate_loc>\w\d+)-(?P<family_num>\w*\d+)-(?P<rep>[A-Z]*\d*)'
    try:
        # print(sample_name, re.search(pat, sample_name).groups())
        # Search for the family name in the sample name
        plate_loc, family_num, rep = re.search(pat, sample_name).groups()
    except AttributeError:
        print('Regex didn\'t match ' + sample_name)
        # If for some reason the sample doesn't match the pattern,
        # it isn't a valid sample
        return None

    # Search for meta data for this family number
    accessions = f'{family_num}-{rep}'
    meta_data = meta_df[meta_df["Accession"] == accessions][meta_df['PlateLocation'] == plate_loc].values
    # If there is one match for family number then return that row
    if len(meta_data) == 1:
        return list(meta_data[0]) + [rep]
    # If there is more than one match for family number then return None
    else:
        return None


def id_chem_areas(chem_info, chem_clf, n_chems):
    '''Return a list of total absorbance areas for each possible
    chemical for all the peaks in a sample.
    '''
    # Generate a row to store the area identified for every chemical
    # group
    row = [0 for _ in range(n_chems)]
    for ret_time, chem_class, total_area in chem_info.values:
        # chem_class is onehot encoded in the chem_clf model.
        # C is first, PP is second.
        if chem_class == 'PP':
            is_C = 0
            is_PP = 1
        else:
            is_C = 1
            is_PP = 0
        # Get the predicted group number (which serves as an index in
        # row) and input the appropriate total absorbance area for
        # that group
        group_i = chem_clf.predict([[ret_time, is_C, is_PP]])[0]  # get the group for this chemical
        row[group_i] = total_area
    return row


# Get the names of all the accumulated .csv files
acc_files = [filename for filename in os.listdir() if filename[:4] == 'acc_']
# Create a list of dataframes of all the accumulated data
acc_dfs = []
for acc_filename in acc_files:
    temp_df = pd.read_csv(acc_filename)
    temp_df['Experiment'] = [acc_filename[4:]] * len(temp_df)  # leave off the 'acc_'
    acc_dfs.append(temp_df)
# Concatenate all the dataframes together
df = pd.concat(acc_dfs, axis=0)

# Get all retention times for normalizing, a 2D array with every value
# in its own inner list
all_ret_times = np.reshape(df['Retention Time'].values, (-1, 1))
# Normalize the retention times
df['Retention Time'] = MinMaxScaler().fit_transform(all_ret_times)

# Load meta data as a dataframe
meta_df = pd.read_csv("supermeta.csv")
# Load the chemical classifier model
chem_clf = joblib.load('chem_grouper.sav')

# Loop through every unique sample name
output_rows = []
for sample_name in set(df["Sample Name"]):
    # Get all rows for this sample name
    sample_rows = df[df["Sample Name"] == sample_name].sort_values(by="Retention Time")
    experiment_name = list(set(sample_rows['Experiment']))[0]  # Extract the experiment name
    # Get the meta data for this sample name
    meta_data = get_meta_data(meta_df, experiment_name, sample_name)
    if not isinstance(meta_data, list):
        # Skip this sample if there was no meta data
        continue

    # Create a dataframe to store some summary data on retention time,
    # chemcial class, and total absorbance area.
    sample_chem_info = sample_rows[['Retention Time', 'Chemical Class']].copy()
    # Create a column of total areas for each wavelength.
    sample_chem_info['Total Areas'] = sample_rows[["218nm Area", "250nm Area", "260nm Area", "330nm Area", "350nm Area"]].sum(axis=1)

    # Get the corrosponding absorbance areas for each identified group
    chem_data = id_chem_areas(sample_chem_info, chem_clf, len(chem_clf.cluster_centers_))

    # Store this row to be added to a df later
    output_rows.append(meta_data + chem_data)

# Create the output dataframe from the output rows
output_df = pd.DataFrame(data=output_rows, columns=(list(meta_df.columns) + ["Repetition Number"] + list(range(len(chem_clf.cluster_centers_)))))

# Save the output df as a csv
output_df.to_csv("final_dataset.csv")
