

import os
import pandas as pd
import sys
import os
from subprocess import call

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
import numpy as np

import random
from scipy.stats import ttest_1samp
from sklearn.externals import joblib
from sklearn.linear_model import LassoLarsCV
from matplotlib.lines import Line2D

random.seed(42)
np.random.seed(42)

from scipy.stats import spearmanr, pearsonr



import sys
sys.path.append('../../')

import common_v2.validation_tools

from common_v2.validation_tools import reps

dataset_filetype = 'loaded_full_dataset'
processed_all_rds = pd.read_pickle(f"../../../data/pieces_new/rocklin_all_rds__all_rds_stability.pkl")



path = "../../../data/"
datasets = ['rd1','rd2', 'rd3', 'rd4']

dfs = {lib:pd.read_table(path+lib+"_stability_scores") for lib in datasets}
all_rds = pd.concat(list(dfs.values())) # except ssm2

#all_rds = all_rds.drop_duplicates(subset='name', keep=False) #dropping duplicate records that have the same name. 
# This step is necessary for the correct join

processed_all_rds.shape


all_rds.shape

all_rds.sequence.unique().shape


train = processed_all_rds[(processed_all_rds.is_train == True) & (processed_all_rds.is_test == False) ]

train.shape


train.sequence.unique().shape



train_sequences_with_annotations = np.intersect1d(train.sequence, all_rds.sequence)
train_sequences_with_annotations.shape



train.shape



train_names = all_rds.set_index('sequence').loc[train.sequence,['name']]



train_names = train_names[~train_names.index.duplicated(keep='first')]


train = train.set_index('sequence')

train.loc[train_names.index,'fold_topology'] = train_names.values
train.fold_topology.isna().value_counts()
train.fold_topology = train.fold_topology.map(lambda x: x.split("_")[0])


train.fold_topology.value_counts().head(15)

topologies = ["EEHEE", "HEEH", "EHEE", "HHH"]


train = train[train.fold_topology.isin(topologies)]




train.shape

models = {}

for top in topologies:
    train_top = train[train.fold_topology == top]
    print(top)
    
    model = LassoLarsCV(
                        fit_intercept = True,
                        normalize = True,
                        n_jobs=-1,
                        max_n_alphas=6000,
                        cv=10
                    )
    
    model.fit(np.asarray(train_top['all_1900'].values.tolist()), train_top['phenotype'].values.astype('float'))
    
    models[top] = model
    print("trained")
    


run_type = 'test'



path = "../../../data/"
datasets = ['rd1','rd2', 'rd3', 'rd4']

dfs = {lib:pd.read_table(path+lib+"_stability_scores") for lib in datasets}
all_rds = pd.concat(list(dfs.values())) # except ssm2

all_rds = all_rds.drop_duplicates(subset='name', keep=False) #dropping duplicate records that have the same name. 
# This step is necessary for the correct join

validate = pd.read_pickle(f"{path}for_rosetta_comparison_rocklin_all_rds_{run_type}_sequences_and_truey.pdpkl")

rock_data = [pd.read_csv(f"{path}rd{i}_relax_scored_filtered_betanov15.sc", sep='\t') for i in [1,2,3,4]]

rock_data = pd.concat([rock_data[x][['sequence', 'description', 'total_score', 'exposed_hydrophobics', 
                                    'buried_np', 
                                    'one_core_each', 
                                    'two_core_each',
                                    'percent_core_SCN',
                                    'buried_minus_exposed', 
                                    'buried_over_exposed', 
                                    'contact_all']] for x in [0,1,2,3]]).set_index('sequence')

validate_meta = all_rds.set_index('sequence').loc[validate.rep].reset_index().copy()

validate_meta.loc[:,'predicted_unirep_fusion_stability'] = np.load(f"{path}rocklin_all_rds__all_rds_stability__all_1900__{run_type}__predictions.npy")

validate_meta.loc[:,'target'] =validate.target

ids_in_common = np.intersect1d(rock_data.description.values,validate_meta.name.dropna().values)

rock_data = rock_data.reset_index().set_index('description')

common_df = validate_meta.set_index('name').loc[ids_in_common]

common_df = common_df.join(rock_data, lsuffix='val', rsuffix='ros_file')

test = processed_all_rds[(processed_all_rds.is_train == False) & (processed_all_rds.is_test == True) ]


test = test.set_index('sequence')

test.shape
common_df.shape


common_df.loc[:,"all_1900"] = test.loc[common_df.sequenceval, 'all_1900'].values



common_df = common_df.reset_index()




results = {}

for top in topologies:
    sliced = common_df[common_df.name.map(lambda x: x.split("_")[0]) == top]
    
    results[top] = {
        'UniRep_all_topo':spearmanr(sliced['predicted_unirep_fusion_stability'],sliced['target'])[0],
        'UniRep_single_topo':spearmanr(models[top].predict(np.asarray(sliced['all_1900'].values.tolist())),sliced['target'])[0],
        'buried_NPSA':spearmanr(sliced['buried_np'],sliced['target'])[0],
        'exposed_NPSA':spearmanr(sliced['exposed_hydrophobics'],sliced['target'])[0]
    }



for top in topologies:
    sliced = common_df[common_df.name.map(lambda x: x.split("_")[0]) == top]
    
    results[top]['Rosetta'] = spearmanr(-sliced['total_score'],sliced['target'])[0]

pd.DataFrame(results).loc[['UniRep_single_topo','Rosetta', 'buried_NPSA', 'exposed_NPSA']]

