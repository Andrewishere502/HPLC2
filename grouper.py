import numpy as np


class Grouper:
    def __init__(self, data):
        """"data -- A list of numbers"""
        # Exclude any nans and sort the data
        self.data = sorted([d for d in data if not np.isnan(d)])
        return
    
    def get_groups(self, max_dif, min_group_size=1):
        """Return a 2D list, where the inner lists contain numbers
        grouped together
        """
        groups = [[self.data[0]]]
        for value in self.data:
            # Get the difference between this value and the previous
            # values.
            diff = abs(value - groups[-1][-1])

            # This value is not more than max_diff from the last value,
            # so add it to the same group
            if diff < max_dif:
                groups[-1].append(value)
            # This value is equal to or more than max_diff from the
            # last value, so it should be in a new group
            else:
                groups.append([value])

        # Return a list of only groups of at least min_group_size
        return [group for group in groups if len(group) >= min_group_size]
