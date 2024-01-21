import os
import pandas as pd

from refiner import Refiner
from grouper import Grouper


def mean(*args):
    """Return the arethmetic mean for a sequence of numbers."""
    return round(sum(args) / len(args), 1)


# Parameter variables for PP
PP_K = 10
PP_MAX_S = 5
PP_MAX_DIF = 0.3
PP_MIN_GROUP_SIZE = 3  # Inclusive

# Parameter variables for C
C_K = 13
C_MAX_S = 10
C_MAX_DIF = 0.5
C_MIN_GROUP_SIZE = 3  # Inclusive



acc_files = [filename for filename in os.listdir()
		if filename[:4] == "acc_"]


pp_ret_times = []
c_ret_times = []
for acc_file in acc_files:
    temp_df = pd.read_csv(acc_file)
    pp_ret_times.extend(list(temp_df[temp_df["Chemical Class"] == "PP"]["Retention Time"]))
    c_ret_times.extend(list(temp_df[temp_df["Chemical Class"] == "C"]["Retention Time"]))

# Refine the Phenylpropanoid column to exclude ambiguous data points
pp_refiner = Refiner(pp_ret_times, PP_K)
pp_ret_times = pp_refiner.remove_outliers(PP_MAX_S)

# Refine the Cardenolide column to exclude ambiguous data points
c_refiner = Refiner(c_ret_times, C_K)
c_ret_times = c_refiner.remove_outliers(C_MAX_S)

# Group the PP points together, split where the next closest
# value to another value is greater than PP_MAX_DIF
pp_grouper = Grouper(pp_ret_times)
pp_groups = pp_grouper.get_groups(PP_MAX_DIF, min_group_size=PP_MIN_GROUP_SIZE)

# Group the C points together, split where the next closest
# value to another value is greater than C_MAX_DIF
c_grouper = Grouper(c_ret_times)
c_groups = c_grouper.get_groups(C_MAX_DIF, min_group_size=C_MIN_GROUP_SIZE)

# Create the chemmeta file
with open("chemmeta.csv", "w") as file:
    # Write the header for the csv file
    file.write("Start,End,Name\n")

    # Write PP groups
    for pp_group in pp_groups:
        # Groups are sorted, take the first and last element to reflect
        # the whole range
        file.write(f"{pp_group[0]},{pp_group[-1]},PP{mean(*pp_group)}\n")

    # Write C groups
    for c_group in c_groups:
        # Groups are sorted, take the first and last element to reflect
        # the whole range
        file.write(f"{c_group[0]},{c_group[-1]},C{mean(*c_group)}\n")
