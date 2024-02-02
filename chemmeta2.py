import random
import os
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib


def numerical_pipeline():
	'''Return an instance of a Pipeline object for scaling and then
	imputing numerical data.
	'''
	steps = (
		('scale', MinMaxScaler()),
		('impute', SimpleImputer())
	)
	return Pipeline(steps)


def gen_n_colors(n_colors):
	'''Return a list of n_unique colors.'''
	colors = []
	for _ in range(n_colors):
		color = (random.random(), random.random(),  random.random())
		colors.append(color)
	return colors


def get_best_n(sil_scores):
	'''Return the n_clusters of the model with the highest silhouette
	score.
	'''
	best_n = 0
	best_score = -1  # The lowest score is -1
	for n_clusters, score in sil_scores.items():
		if score > best_score:
			best_n = n_clusters
			best_score = score
	return best_n


# Load the accumulated data filenamess for each experiment
acc_filenames = [filename for filename in os.listdir() 
				if filename[:4] == 'acc_']

# Create a list of dataframes of all the accumulated data
acc_dfs = []
for acc_filename in acc_filenames:
	acc_dfs.append(pd.read_csv(acc_filename))

# Concatenate all the dataframes together
df = pd.concat(acc_dfs, axis=0)

# Create the preprocessing pipeline
# preprocessing = ColumnTransformer([
# 	('num', numerical_pipeline(), ['Retention Time', '218nm Area', '250nm Area',
# 									'260nm Area', '330nm Area', '350nm Area']),
# 	('cat', OneHotEncoder(), ['Chemical Class'])
# 	], remainder='drop'  # drop sample name for training dataset
# )
preprocessing = ColumnTransformer([
	('num', numerical_pipeline(), ['Retention Time']),
	('cat', OneHotEncoder(), ['Chemical Class'])
	], remainder='drop'  # drop sample name for training dataset
)
preprocessed_df = preprocessing.fit_transform(df)

# Create many KMeans models looking for the best one indicated by
# the highest silhouette score.
sil_scores = {}
for n_clusters in range(10, 100): 
	km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=2)
	sil_score = silhouette_score(preprocessed_df, km.fit_predict(preprocessed_df))
	sil_scores.update({n_clusters: sil_score})

# Visualize which n has the highest silhouette score
plt.plot(sil_scores.keys(), sil_scores.values())
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Create the best kmeans model
n_clusters = get_best_n(sil_scores)
km_best = KMeans(n_clusters=n_clusters, n_init='auto', random_state=2)
group_labels = km_best.fit_predict(preprocessed_df)

# Label the instances by group # assigned by the KMeans model
df_labeled = df.copy()
df_labeled['Group #'] = group_labels

# Check the purity of each group, groups should really only have all
# phenylpropanoids or all cardenolides.
max_group = df_labeled['Group #'].max()  # Get the highest group number
chem_data = []
for group_num in range(max_group + 1):
	# Grab a dataframe with only all instances that are part of this
	# group number.
	group_num_df = df_labeled[df_labeled['Group #'] == group_num]

	# Get a list of all chemical classes that are part of this group
	# number.
	chem_classes = list(group_num_df['Chemical Class'])

	# Calculate how much of this group is composed of phenylpropanoids
	PP_rate = len([chem_class for chem_class in chem_classes
					if chem_class == 'PP']) / len(chem_classes)
	# Save some metadata on this group
	row = [
		group_num,  # Save the group number
		len(chem_classes),  # Save the number of chemicals in this group
	   	PP_rate,  # Save the proportion of phenylpropanoids in this group
	   	round(group_num_df['Retention Time'].min(), 2),  # Save the minimum retention time in this group
	   	round(group_num_df['Retention Time'].max(), 2),  # Save the maximum retention time in this group
	   	round(group_num_df['Retention Time'].median(), 2)  # Save the median retention time in this group
	]
	chem_data.append(row)


# Save stats of the chemical group inside the model
output_df = pd.DataFrame(chem_data, columns=['Group Number', 'Count',
											 'Phenylpropanoid Purity',
											 'Retention Min', 'Retention Max', 'Retention Median',])
output_df.to_csv('chemmeta.csv', index=False, index_label=False)
# Save the KMeans clustering model
joblib.dump(km_best, 'chem_grouper.sav')
