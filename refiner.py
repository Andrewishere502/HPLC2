import numpy as np


class Refiner:
    def __init__(self, data, k):
        """"data -- A list of numbers"""
        self.data = data
        self.scores = self.score_data(data, k)
        return
    
    @staticmethod
    def score_point(point_i, data, k):
        # point_i is the index of the point to compare in data
        # data is a list of all points
        # k is the number of nearest neighbors to compare point to

        point = data[point_i]
        distances = []
        for neighbor_i, neighbor in enumerate(data):
            # Prevent comparing point to itself
            if point_i == neighbor_i:
                continue
            # Get distance between points
            d = abs(neighbor - point) ** 2
            # Add this distance to scores
            distances.append(d)

        # Sum scores of closest k neighbors
        k_total = sum(sorted(distances)[:k])

        # Take kth root of k_total to get score
        score = k_total / k
        return score

    def score_data(self, data, k):
        # Prevent k values greater than the actual number of points a
        # given point can possibly be compared to.
        if k > len(data) - 1:
            raise ValueError(f"k = {k} is invalid for only {len(data)} data points")

        scores = []
        for point_i, _ in enumerate(data):
            score = self.score_point(point_i, data, k)
            scores.append(score)
        return scores
    
    def remove_outliers(self, max_s):
        # max_s is the maximum allowed score for a point
        # to be returned. Exclusive.

        no_outliers = []
        for i, point in enumerate(self.data):
            if self.scores[i] < max_s:
                no_outliers.append(point)
            else:
                no_outliers.append(np.NaN)
        return no_outliers
