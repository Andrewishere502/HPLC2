import re

import pandas as pd


def get_meta_data(meta_df, sample_name):
    """Return the meta data for a sample_name as a list with the
    repetition number tacked on to the end. List values are ordered
    corresponding to the columns in meta_df.
    """
    # Each sample name should follow this pattern
    pat = r"5wk_(?P<family_num>\d+)-(?P<rep>\d+)"
    try:
        # Search for the family name in the sample name
        family_num, rep = map(int, re.search(pat, sample_name).groups())
    except AttributeError:
        # If for some reason the sample doesn't match the pattern,
        # it isn't a valid sample
        return None

    # Search for meta data for this family number
    meta_data = meta_df[meta_df["FamilyNumber"] == family_num].values
    # If there is one match for family number then return that row
    if len(meta_data) == 1:
        return list(meta_data[0]) + [rep]
    # If there is more than one match for family number then return None
    else:
        return None
    

def get_chem_names(chem_df, ret_times, total_areas):
    """Return a list of booleans corresponding to each chem name in
    chem_df.
    """
    chem_names = []
    for start_ret, end_ret, chem_name in chem_df.values:
        area = 1
        for i, ret_time in enumerate(ret_times):
            if ret_time >= start_ret and ret_time <= end_ret:
                area = total_areas[i]
        chem_names.append(area)
    return chem_names


# Load meta and chemical dataframe
meta_df = pd.read_csv("supermeta.csv")
chem_df = pd.read_csv("chemmeta.csv")


output_rows = []
acc_files = ["acc_SUMR2023-07-1108-35-49.csv",
             "acc_SUMR2023-07-1116-38-47.csv",
             "acc_SUMR2023-07-1811-01-20.csv"]
for acc_file in acc_files:
    # Load the accumulated csv file
    df = pd.read_csv(acc_file)

    # Loop through every unique sample name
    for sample_name in set(df["Sample Name"]):
        # Get all rows for this sample name
        sample_rows = df[df["Sample Name"] == sample_name].sort_values(by="Retention Time")

        # Get the meta data for this sample name
        meta_data = get_meta_data(meta_df, sample_name)
        if not isinstance(meta_data, list):
            # Skip this sample if there was no meta data
            continue
        
        # Create a list of retention times
        ret_times = list(sample_rows["Retention Time"])
        # Create a list of total areas, indices correspond to ret_times
        total_areas = sample_rows[["218nm Area", "250nm Area", "260nm Area", "330nm Area", "350nm Area"]].sum(axis=1)
        # Get the chem data for this sample name
        chem_data = get_chem_names(chem_df, ret_times, list(total_areas))
        
        # Store this row to be added to a df later
        output_rows.append(meta_data + chem_data)

# Create the output dataframe from the output rows
output_df = pd.DataFrame(data=output_rows, columns=(list(meta_df.columns) + ["Repetition Number"] + list(chem_df["Name"])))

# Save the output df as a csv
output_df.to_csv("final_dataset.csv")
