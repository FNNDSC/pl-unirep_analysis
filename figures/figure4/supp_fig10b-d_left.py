"""

Reproduce Supp. Fig. 10b,c,d left-hand column (test set)

Test set. Plot the transfer results for the test set from variant effect prediciton (Fowler,2018), Stability Deep Mutational Scanning on Natural and de novo designs (Rocklin, 2017) and extrapolation from Stability Deep Mutational Scanning on Natural and de novo designs (Rocklin, 2017) (Methods).
"""



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import random
import os
import pickle
from sklearn.externals import joblib
# To allow imports from common directory
sys.path.append('../../')
from common.plot_style_utils import set_pub_plot_context, rep_names, main_text_rep_names, save_for_pub, label_point

%matplotlib inline
random.seed(42)
np.random.seed(42)

set_pub_plot_context()
sns.palplot(sns.color_palette())
with open("../../data/transfer_all_rep_results_test.pkl", "rb") as p:
    scores = joblib.load(p)
display(scores.keys())



scores['fowler_consistent_single_UBI']



scores['fowler_consistent_single_UBI'].sort_values(by='transfer_ratio_avg')

fowler = scores['fowler_consistent_single_UBI'].loc[
    ['avg_hidden', 'RGN_avg_hidden', 'RGN','arnold_uniform_4_1',"tfidf_3grams"], :
]
fowler

set_pub_plot_context(context="notebook")
fig = plt.figure(figsize=(3,3), dpi=250)
ax = fig.add_subplot(111,
                    xlabel="In-Domain Ratio",
                    ylabel="Transfer Ratio")
palette = sns.color_palette()
color_idxs = [0,5,1,2,3]
colors = [palette[i] for i in color_idxs]
names = [main_text_rep_names[o] for o in fowler.index.values.tolist()]
names[3] = "Best Arnold"
names[4] = "Our Best Baseline"
for i, (_, row) in enumerate(fowler.iterrows()):
    ax.scatter(x=row['indomain_ratio_avg'], 
                y=row['transfer_ratio_avg'], 
                c=colors[i])

    
label_point(pd.Series(fowler.indomain_ratio_avg.values),
            pd.Series(fowler.transfer_ratio_avg.values),
            pd.Series(
                names
            ),ax, fontsize=12)
save_for_pub(fig, path="./img/b_test", dpi=250)



ssm2 = scores['rocklin_ssm2'].loc[
    ['avg_hidden', 'RGN_avg_hidden', 'RGN','arnold_original_3_7',"simple_freq_and_len"], :
]
ssm2

set_pub_plot_context(context="notebook")
fig = plt.figure(figsize=(3,3), dpi=250)
ax = fig.add_subplot(111,
                    xlabel="In-Domain Ratio",
                    ylabel="Transfer Ratio")
palette = sns.color_palette()
color_idxs = [0,5,1,2,3]
colors = [palette[i] for i in color_idxs]
names = [main_text_rep_names[o] for o in ssm2.index.values.tolist()]
names[3] = "Best Arnold"
names[4] = "Our Best Baseline"
for i, (_, row) in enumerate(ssm2.iterrows()):
    ax.scatter(x=row['indomain_ratio_avg'], 
                y=row['transfer_ratio_avg'], 
                c=colors[i])

    
label_point(pd.Series(ssm2.indomain_ratio_avg.values),
            pd.Series(ssm2.transfer_ratio_avg.values),
            pd.Series(
                names
            ),ax, fontsize=12)
save_for_pub(fig, path="./img/c_test", dpi=250)

extrap_df = pd.read_csv("../../data/for_figure_rocklin_ssm2_to_remote_transfer_results_test.csv", index_col=0)
extrap_df
extrap_df.sort_values(by='transfer_ratio_avg')

ssm2_extrap = extrap_df.loc[
    ['avg_hidden', 'RGN_avg_hidden', 'RGN','arnold_random_3_7',"simple_freq_and_len"], :
]
ssm2_extrap

set_pub_plot_context(context="notebook")
fig = plt.figure(figsize=(3,3), dpi=250)
ax = fig.add_subplot(111,
                    xlabel="In-Domain Ratio",
                    ylabel="Transfer Ratio")
palette = sns.color_palette()
color_idxs = [0,5,1,2,3]
colors = [palette[i] for i in color_idxs]
names = [main_text_rep_names[o] for o in ssm2.index.values.tolist()]
names[3] = "Best Arnold"
names[4] = "Our Best Baseline"
for i, (_, row) in enumerate(ssm2_extrap.iterrows()):
    ax.scatter(x=row['indomain_ratio_avg'], 
                y=row['transfer_ratio_avg'], 
                c=colors[i])
    
label_point(pd.Series(ssm2_extrap.indomain_ratio_avg.values),
            pd.Series(ssm2_extrap.transfer_ratio_avg.values),
            pd.Series(
                names
            ),ax, fontsize=12)
save_for_pub(fig, path="./img/d_test", dpi=250)



