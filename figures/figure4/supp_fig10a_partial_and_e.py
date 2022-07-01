"""

Reproduce Supp. Fig. 10a (top line) and Supp. Fig. 10e

Compute a multi-dimensional scaling of the raw sequences (Levenshtein distance) deep mutational scanning stability data from Rocklin (2017). Use the first principle coordinate to make the upper line (labelled sequence PC1) in Supp. Fig. 10a
"""
import numpy as np
import Levenshtein
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.manifold import MDS
import pandas as pd
import sys
import random
import os
import Levenshtein
import fastcluster as fc
import scipy.cluster.hierarchy as hier
import pickle
# To allow imports from common directory
sys.path.append('../../')
from common.plot_style_utils import set_pub_plot_context, rep_names, save_for_pub, label_point
from common.embedding_tools import lev_dist_matrix, lev_sim_matrix

%matplotlib inline
random.seed(42)
np.random.seed(42)

wt_ssm2 = pd.read_csv("../../data/just_wt_ssm2_sequences.csv", index_col=0)
wt_ssm2



wt_ssm2 = wt_ssm2.loc[:,["name", "sequence"]]
wt_ssm2



wt_ssm2 = wt_ssm2.reset_index()

lev_d = lev_dist_matrix(wt_ssm2.sequence.values)
lev_d
mds = MDS(n_components=2, n_init=8, verbose=10, n_jobs=4, random_state=42, dissimilarity='precomputed')
mds.fit(lev_d)



print(mds.embedding_.shape)
mds.embedding_

components = mds.embedding_



# Compute the string median and also vidualize this
median_seq = Levenshtein.median(wt_ssm2['sequence'].values.tolist())
median_seq



# Try to make this an even better median
median_improved = Levenshtein.median_improve(median_seq, wt_ssm2['sequence'].values.tolist())
median_improved



# How many times does this work?
median_improved2 = Levenshtein.median_improve(median_improved, wt_ssm2['sequence'].values.tolist())
median_improved2

# I'm not sure what's going on here, let me just iterate on this for a while
working_median = median_improved2
for i in range(1000):
    if i % 100 == 0:
        print(f"Iter {i} has working_median {working_median}")
    working_median = Levenshtein.median_improve(working_median, wt_ssm2['sequence'].values.tolist())
    
# cool, looks like thats a real median. Lets recompute lev distance
# including that point.
new_sequences = np.array(
    wt_ssm2.sequence.values.tolist() + [working_median]
)
print(new_sequences)
lev_d = lev_dist_matrix(new_sequences)
lev_d
mds = MDS(n_components=2, n_init=8, n_jobs=4, random_state=42, dissimilarity='precomputed')
mds.fit(lev_d)



components = mds.embedding_

# now plot again but color these
set_pub_plot_context(context="notebook")
name_of_test = ['villin']
name_of_train = wt_ssm2[~wt_ssm2['name'].isin(name_of_test)]['name']
test_idxs = wt_ssm2[wt_ssm2['name'].isin(name_of_test)].index.values
train_idxs = wt_ssm2[~wt_ssm2['name'].isin(name_of_test)].index.values
fig = plt.figure(figsize=(6,6), dpi=250)
ax = fig.add_subplot(111)
x = components[train_idxs,0]
y = components[train_idxs,1]
color = mpl.colors.rgb2hex(sns.color_palette()[0])
sns.regplot(x=x, y=y, fit_reg=False, ax=ax, color=color, label="Train")
label_point(pd.Series(x),pd.Series(y+.75),
            pd.Series(np.array(name_of_train)).map(lambda x: x.split(".")[0])
            ,ax, fontsize=10)

x = components[test_idxs,0]
y = components[test_idxs,1]
color = mpl.colors.rgb2hex(sns.color_palette()[1])
sns.regplot(x=x, y=y, fit_reg=False, ax=ax, color=color, label="Test")
label_point(pd.Series(x),pd.Series(y+.75),
            pd.Series(np.array(name_of_test)).map(lambda x: x.split(".")[0])
            ,ax, fontsize=10)
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax.legend(loc="lower left",fancybox=True, frameon=True)
save_for_pub(fig, path="./img/supp_fig10e", dpi=250)


# now plot again but color these
y_bump = -.001
set_pub_plot_context(context="notebook")
name_of_reference = ['hYAP65']
name_of_test = ['villin']
name_of_train = ['HEEH_rd3_0872.pdb','HEEH_rd3_0223.pdb','HEEH_rd3_0726.pdb',
                 'EEHEE_rd3_1716.pdb',
                 "EHEE_rd2_0005.pdb", "EHEE_0882.pdb"]
test_idxs = wt_ssm2[wt_ssm2['name'].isin(name_of_test)].index.values
train_idxs = wt_ssm2[wt_ssm2['name'].isin(name_of_train)].index.values
ref_idxs = wt_ssm2[wt_ssm2['name'].isin(name_of_reference)].index.values
fig = plt.figure(figsize=(3,3), dpi=250)
ax = fig.add_subplot(111)
x = components[train_idxs,0].tolist()

y = np.zeros(len(train_idxs)).tolist()

color = mpl.colors.rgb2hex(sns.color_palette()[0])
sns.regplot(x=np.array(x), y=np.array(y), fit_reg=False, ax=ax, color=color, label="Train")


x = components[test_idxs,0].tolist()
y = np.zeros(len(test_idxs)).tolist()
color = mpl.colors.rgb2hex(sns.color_palette()[1])
sns.regplot(x=np.array(x), y=np.array(y), fit_reg=False, ax=ax, color=color, label="Test")

x = components[ref_idxs,0].tolist()
y = np.zeros(len(ref_idxs)).tolist()
color = 'grey'
sns.regplot(x=np.array(x), y=np.array(y), fit_reg=False, ax=ax, color=color, label="Distant Reference")
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
ax.legend(loc="lower left",fancybox=True, frameon=True)
save_for_pub(fig, path="./img/supp_fig10a_top", dpi=250)



