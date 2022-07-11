"""
Analysis: unsupervised agglomerative Euclidean-distance based clustering of Oxbench and Homstrad, compared to all baselines.

This notebook was used to creat the clustering results files used to plot. By defualt, the results are not saved (overwriting the pre-packaged result) unless you uncommont the save lines.

This is computationally intensive and takes tens of minutes-hours to run on a 16G laptop.

"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as hier
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import random
import os
import Levenshtein
import fastcluster as fc
# To allow imports from common directory
sys.path.append('../../')
from common.embedding_tools import lev_dist_matrix, lev_sim_matrix

%matplotlib inline
random.seed(42)
np.random.seed(42)

homstrad = pd.read_pickle("../../data/homstrad_w_baseline.pkl")
homstrad.phenotype.value_counts()
homstrad.columns

oxbench = pd.read_pickle("../../data/oxbench_w_baseline.pkl")
oxbench.phenotype.value_counts()
"""


Ok the pseudo code here is as follows:

Oxbench

    Let the phenotype (family name) be the class label
    Altogether (as a single clustering):
        Perform Agglomerative clustering on reps, baselines, and on Levenshtein
        Compute clustering accuracy using sklearn metrics (various)

Homstrad

    Assign each phenotype name a unique class index integer
    Altogether (as a single clustering):
        Perform Agglomerative clustering on reps, baselines, and on Levenshtein
        Compute clustering accuracy using sklearn metrics (various)

Should save the metrics of the result in a .csv for figure making later.

"""
column_idxs = list(range(len(oxbench.columns)))
# filter out Nans from RGN reps
filtered = oxbench[np.isnan(np.asarray(oxbench.RGN.values.tolist())).sum(axis=1) == 0]
X = filtered.iloc[:,[0] + column_idxs[6:]]
display(X)
labels = filtered['phenotype']
display(labels)
ox_results = cluster(X, labels, n_clusters=len(labels.unique()))
display(ox_results)

to_plot = ox_results.loc["ag_ari", :]
to_plot.index
print("Oxbench Performance, Adjusted Rand Index")
to_plot.sort_values(ascending=False)

to_plot = ox_results.loc["ag_fmi", :]
to_plot.index
print("Oxbench Performance, Fowlkes Mallows Index")
to_plot.sort_values(ascending=False)

to_plot = ox_results.loc["ag_ami", :]
to_plot.index
print("Oxbench Performance, Adjusted Mutual-Information Score")
to_plot.sort_values(ascending=False)

ox_results = ox_results.T
ox_results.columns = ["Adjusted Rand Index", "Fowlkes Mallows Index", "Adjusted Mutual Information"]

# Save results for later
# ox_results.to_csv("../../data/oxbench_agglom_results.csv")

display(ox_results.sort_values(by="Fowlkes Mallows Index", ascending=False))



# First I need to get unique indices for all family names
display(homstrad)


filtered_homstrad = homstrad[np.isnan(np.asarray(homstrad.RGN.values.tolist())).sum(axis=1) == 0]

print(filtered_homstrad.shape)
print(homstrad.shape)
encoder = LabelEncoder()
encoder.fit(filtered_homstrad['phenotype'])
labels = encoder.transform(filtered_homstrad['phenotype'])
print(labels)
print(encoder.inverse_transform(labels))



# Nice! Now cluster with scores (later I will agglo cluster with dendro)
column_idxs = list(range(len(filtered_homstrad.columns)))
X = filtered_homstrad.iloc[:,[0] + column_idxs[6:]]
display(X)
display(labels) # from previous cell
hom_results = cluster(X, labels, n_clusters=len(np.unique(labels)))



hom_results



to_plot = hom_results.loc["ag_ari", :]
to_plot.index
print("Homstrad Performance, Adjusted Rand Index")
to_plot.sort_values(ascending=False)

to_plot = hom_results.loc["ag_fmi", :]
to_plot.index
print("Homstrad Performance, Fowlkes Mallows Index")
to_plot.sort_values(ascending=False)

to_plot = hom_results.loc["ag_ami", :]
to_plot.index
print("Homstrad Performance, Adjusted Mutual Information Score")
to_plot.sort_values(ascending=False)



hom_results = hom_results.T
hom_results.columns = ["Adjusted Rand Index", "Fowlkes Mallows Index", "Adjusted Mutual Information"]



# Save results for later
# hom_results.to_csv("../../data/homstrad_agglom_results.csv")



hom_results.sort_values(by="Fowlkes Mallows Index", ascending=False)




