

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor

## COMPUTING THE NEAREST NEIGHBOR BASELINE

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
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestRegressor

random.seed(42)
np.random.seed(42)

from scipy.stats import spearmanr, pearsonr
from sklearn.externals import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import sys
sys.path.append('./UniRep-analysis/')

import common_v2.validation_tools
import scipy.stats

test_seqs = {}
train_seqs = {}
run_type = 'test'

from common_v2.validation_tools import reps, subsets

dataset_filetype = 'loaded_full_dataset'
path = pd.read_pickle(f"../data/pieces_new/rocklin_ssm2__full.pkl")

for s in subsets['rocklin_ssm2']:

    rep='sequence'
    train, validate, test = common_v2.validation_tools.get_tvt(path,
                                                             'rocklin_ssm2',
                                                             s,
                                                             rep, 
                                       dataset_filetype=dataset_filetype, 
                                       verbose=False,

                                                                modifiers=[common_v2.validation_tools.tvt_modifier_baseline_reps,
                                                                        common_v2.validation_tools.tvt_modifier_return_mean]
                            )

    if run_type == 'test':
        validate = test
        
        
    test_seqs[s]= validate
    train_seqs[s]=train

df=pd.read_csv('./ssm2_stability_scores',delim_whitespace=True)

wildtypes = df[df.pos == 0].set_index('name')[['consensus_stability_score', 'sequence']]


aas='ADEFGHIKLMNPQRSTVWY'

aa_diff_for_each_wildtype = {}

for wt in wildtypes.index:
    
    aa_diff={}

    for i, aa1 in enumerate(aas[:-1]):
        for j, aa2 in enumerate(aas[i+1:]):
            aa_diff[(aa1, aa2)] = []
    
    print (wt)
    subdf=df[(df['my_wt'] == wt) & (df.sequence.map(lambda x: x in train_seqs[wt + '_ssm2_stability'].rep.tolist()))]
    print(subdf.shape)
    
    wtss = wildtypes.loc[wt, 'consensus_stability_score']
    
    for pos in list(set([x for x in subdf['pos'] if x != 0])):
        subdf2=subdf.query('pos == %s' % pos)
        for mut, wt_aa, ss in zip(subdf2['mut'], subdf2['wt_aa'], subdf2['consensus_stability_score']):
            aa_diff[tuple(sorted([mut, wt_aa]))].append(abs(wtss - ss))
        for mut1, ss1 in zip(subdf2['mut'], subdf2['consensus_stability_score']):
            for mut2, ss2 in zip(subdf2['mut'], subdf2['consensus_stability_score']):
                if mut1 < mut2:
                    aa_diff[tuple(sorted([mut1, mut2]))].append(abs(ss1 - ss2))
                    
    aagrid=np.zeros((19,19))
    for i, aa1 in enumerate(aas):
        for j, aa2 in enumerate(aas):
            if i != j: aagrid[i,j] = np.average(aa_diff[tuple(sorted([aa1, aa2]))])
        
                    
    aa_diff_for_each_wildtype[wt] = aagrid

wildtypes['pearson_nn'] = 0
for wt in wildtypes.index:

    train = df[df.sequence.map(lambda x: x in train_seqs[wt+ '_ssm2_stability'].rep.tolist())]
    test = df[df.sequence.map(lambda x: x in test_seqs[wt+ '_ssm2_stability'].rep.tolist())]

    aagrid = aa_diff_for_each_wildtype[wt]

    out = []
    for pos, wt_aa, mut, ss in zip(test['pos'], test['wt_aa'], test['mut'], test['consensus_stability_score']):
        #print(aas, mut)
        #print(aas.index(mut))
        
        if mut == 'na':
            out.append((pos, mut, newaa, ss, wildtypes.loc[wt,'consensus_stability_score']))
        else:
            for newaa in (np.array([x for x in aas])[np.argsort(aagrid[aas.index(mut)])[1:]]):

                nndf=train.query('pos == %s & mut == "%s"' % (pos, newaa))
                if len(nndf) == 1:
                    out.append((pos, mut, newaa, ss, nndf['consensus_stability_score'].values[-1]))
                    break
    out=pd.DataFrame(out)
    out.columns=['pos','mut','nnaa','consensus_stability_score','nn_consensus_stability_score']
    
    wildtypes.loc[wt,'pearson_nn'] = scipy.stats.pearsonr(test_seqs[wt+'_ssm2_stability'].target, out.nn_consensus_stability_score)[0]
    
    
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
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestRegressor

random.seed(42)
np.random.seed(42)

from scipy.stats import spearmanr, pearsonr
from sklearn.externals import joblib

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import sys
sys.path.append('../')

import common_v2.validation_tools

from common_v2.validation_tools import reps, subsets

dataset_filetype = 'loaded_full_dataset'
path = pd.read_pickle(f"../data/pieces_new/rocklin_ssm2__full.pkl")  



def id_mutations(wt, mut_seq):
    # identify mutations in same length wt and seq2
    n_mut = 0
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            n_mut+=1
    return n_mut

def return_mutation_position(wt, mut_seq):
    # identify mutations in same length wt and seq2
    muts = []
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            return i
    raise



from common_v2.validation_tools import metrics


run_type = 'test'

class simple_baseline_mutation_effect_model:
    def train(self, train, wt_seq, length):
        
        train["mutated_position"] = train.rep.map(lambda mut_seq: return_mutation_position(wt_seq, mut_seq))
        
        train.target = np.float16(train.target)
        
        avg_effects = train.groupby(['mutated_position']).target.mean()
        
        if len(avg_effects) != len(wt_seq):
            print("not all positions mutated in training data")
            
        self.avg_effects = avg_effects
        
    def predict(self, validate):
        
        return validate.rep.map(lambda mut_seq: self.avg_effects[return_mutation_position(wt_seq, mut_seq)]).values

subset_results = {}

for s in subsets['rocklin_ssm2']:

    rep='sequence'
    train, validate, test = common_v2.validation_tools.get_tvt(path,
                                                             'rocklin_ssm2',
                                                             s,
                                                             rep, 
                                       dataset_filetype=dataset_filetype, 
                                       verbose=False,

                                                                modifiers=[common_v2.validation_tools.tvt_modifier_baseline_reps,
                                                                        common_v2.validation_tools.tvt_modifier_return_mean]
                            )

    if run_type == 'test':
        validate = test
    
    print(s)
   
    length = train.rep.map(len).value_counts().index[0]
    wt_seq = "".join([train.rep.map(lambda x: x[i]).value_counts().index[0] for i in range(length)])
    
    print(length)
    
    model = simple_baseline_mutation_effect_model()
    
    train = train[train.rep != wt_seq]
    validate = validate[validate.rep != wt_seq]

    model.train(train, wt_seq, length)

    predictions = model.predict(validate)
    print(len(predictions), len(validate))
    
    
    results = pd.Series()

    for metric_name in metrics.keys():
        #print(validate['target'].values)
        #print(predictions)
        results.loc[metric_name] = metrics[metric_name](validate['target'],
                   predictions)
    
    our_results  = pd.read_csv(f"../data/results/rocklin_ssm2__{s}__all_1900__{run_type}__regression_results.csv", header=None, index_col=0)
    
    subset_results[s] = {"avg_pos_effect":results['pearson_r'], "UniRep":our_results.loc['pearson_r'].values[0]}



df = pd.DataFrame(subset_results)
df.T.plot(kind='barh')

knn_results = {}

for s in subsets['rocklin_ssm2']:

    rep='all_1900'
    train, validate, test = common_v2.validation_tools.get_tvt(path,
                                                             'rocklin_ssm2',
                                                             s,
                                                             rep, 
                                                             dataset_filetype=dataset_filetype, 
                                                             verbose=False,
                                                             modifiers=[common_v2.validation_tools.tvt_modifier_baseline_reps,
                                                                        common_v2.validation_tools.tvt_modifier_return_mean]
                            )

    if run_type == 'test':
        validate = test
    
    print(s)

    model = KNeighborsRegressor(n_neighbors=10, metric='euclidean', weights='distance')

    #
    y_train = train.target.values.tolist()

    model.fit(
        np.asarray(train.rep.values.tolist()), 
        y_train)
    
    try:
        print(model.best_params_)
    except:
        pass

    predictions = model.predict(np.asarray(validate.rep.values.tolist()))

    results = pd.Series()

    for metric_name in metrics.keys():
        #print(validate['target'].values)
        #print(predictions)
        results.loc[metric_name] = metrics[metric_name](validate['target'],
                   predictions)
    
    
    knn_results[s] = results['pearson_r']
   
 for k in subset_results.keys():
    subset_results[k]['KNN'] = knn_results[k]

 df = pd.DataFrame(subset_results)
 df  
 
 

df.index = ['UniRep+KNN', 'UniRep+Lasso', 'Average Positional Effect']

plt.set_cmap('Set2')


wildtypes['pearson_nn'].values
df.loc['Nearest Neighbor', wildtypes['pearson_nn'].index.map(lambda x: x+"_ssm2_stability")] = wildtypes['pearson_nn'].values


joblib.dump(df, "ssm2_baseline_analysis_subset_results_df_PROPER.joblibpkl")


import palettable as pal



palette = pal.cartocolors.qualitative.Safe_10.get_mpl_colormap()


df.loc[['UniRep+Lasso', 'Nearest Neighbor', 'Average Positional Effect']].T.plot(kind='barh', figsize=(7,10), cmap=palette)
plt.legend(loc='upper center', bbox_to_anchor=(1.37, 0.55), fancybox=True, shadow=True)
plt.xlabel('Pearson r')
plt.title('Positional effect baseline comparison for 17 proteins in the DMS stability prediction task')



df.loc[['UniRep+Lasso', 'Nearest Neighbor', 'UniRep+KNN', 'Average Positional Effect'],'EEHEE_rd3_1702.pdb_ssm2_stability'].plot(kind='barh', cmap=palette)
plt.xticks(rotation=90)
plt.xlabel("Pearson r")
plt.title("KNN top model performance on EEHEE_rd3_1702 de novo design")


