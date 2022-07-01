

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

random.seed(42)
np.random.seed(42)

from scipy.stats import spearmanr, pearsonr



ssm2 = pd.read_pickle("../../../data/pieces_new/rocklin_ssm2__full.pkl")


hyap = ssm2[ssm2.phenotype_name.str.contains("hYAP65")]


def id_mutations(wt, mut_seq):
    # identify mutations in same length wt and seq2
    n_mut = 0
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            n_mut+=1
    return n_mut

def return_mutations(wt, mut_seq):
    # identify mutations in same length wt and seq2
    muts = []
    for i,letter in enumerate(list(mut_seq)):
        if letter != wt[i]:
            return f"{wt[i]}{i+1}{mut_seq[i]}"
    raise
    


"".join([hyap.sequence.map(lambda x: x[i]).value_counts().index[0] for i in range(46)])



hyap.sequence.map(lambda mut_seq: id_mutations('FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM', mut_seq)).value_counts()



for prot in ssm2.phenotype_name.unique():
    print(prot)
    prot_subset = ssm2[ssm2.phenotype_name == prot]
    wt = "".join([prot_subset.sequence.map(lambda x: x[i]).value_counts().index[0] for i in range(len(prot_subset.sequence.iloc[0]))])
    print(prot_subset.sequence.map(lambda mut_seq: id_mutations(wt, mut_seq)).value_counts())

hyap_muts = hyap[hyap.sequence != "FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM"]


return_mutations("FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM", "FEIPADVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM")



for x in hyap_muts.sequence.map(lambda mut_seq: return_mutations('FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM', mut_seq)):
    print(f"{x};")

hyap.loc[hyap_muts.index, 'mutation'] = hyap_muts.sequence.map(lambda mut_seq: return_mutations('FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM', mut_seq))


from io import StringIO



hyap_muts.loc[:,'strum_pred'] = strum_res.iloc[:,-1].values



test = hyap_muts[hyap_muts.is_test == True]




unirep_predictions = np.load(
    "../../../data/predictions/rocklin_ssm2__hYAP65_ssm2_stability__all_1900__test__predictions.npy"
)



from scipy.stats import spearmanr



spearmanr(unirep_predictions, test['phenotype'])



spearmanr(test['strum_pred'],
          test['phenotype'])


def compute_std(predictions, real_values):

    return pd.Series([
        spearmanr(
            predictions[intersec_ids_sample],
            real_values[intersec_ids_sample])[0] for intersec_ids_sample in [
                np.random.choice(
                            np.arange(real_values.shape[0]), size = np.int(real_values.shape[0]/2), replace=False) for i in range(30)
                    ] 
    ]).std()


from scipy.stats import ttest_ind_from_stats

ttest_ind_from_stats(mean1=spearmanr(unirep_predictions, test['phenotype'].values)[0], 
                     std1=compute_std(unirep_predictions, test['phenotype'].values), 
                     nobs1=30,
                     mean2=spearmanr(test['strum_pred'].values,test['phenotype'].values)[0], 
                     std2 =compute_std(test['strum_pred'].values,test['phenotype'].values), 
                     nobs2=30)


def get_wt_and_print_test_set_mutations(phen_name):
    hyap = ssm2[ssm2.phenotype_name.str.contains(phen_name)]
    length = hyap.sequence.map(len).value_counts().index[0]
    wt = "".join([hyap.sequence.map(lambda x: x[i]).value_counts().index[0] for i in range(length)])
    print(wt)
    hyap_muts = hyap[hyap.sequence != wt]
    
    test = hyap_muts[hyap_muts.is_test == True]
    
    for x in test.sequence.map(lambda mut_seq: return_mutations(wt, mut_seq)):
        print(f"{x};")

return_mutations("FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM", "FEIPDDVPLPAGWEMAKTSSGQRPFKNHIDQTTTWQDPRKAMLSQM")


ssm2.phenotype_name.value_counts()




get_wt_and_print_test_set_mutations('villin')



villin_res = pd.read_table(StringIO("""
Position	Wild-type	mutant type	ddG
21	A	P	-0.64
21	A	T	0.25
21	A	W	0.41
23	A	G	-0.02
23	A	H	0.78
23	A	P	-0.17
13	A	F	1.11
13	A	G	0.46
13	A	S	0.42
8	D	K	-0.17
8	D	Y	0.13
10	D	F	-0.71
10	D	Q	-0.53
10	D	T	-0.83
36	E	P	-0.54
9	E	F	0.6
15	F	H	-0.66
22	F	D	-2.49
22	F	H	-2.51
22	F	N	-1.69
40	F	I	-0.85
40	F	M	-0.35
11	F	L	-2.53
11	F	R	-1.48
16	G	D	0.48
16	G	K	0.22
16	G	Q	0.03
16	G	W	-0.26
38	G	I	0.03
38	G	P	-0.96
38	G	W	0.18
34	K	M	-0.23
34	K	Y	-0.71
35	K	G	0.19
35	K	M	0.71
35	K	N	0.58
37	K	M	0.94
6	L	S	-0.82
25	L	A	-1.53
25	L	M	-0.25
27	L	A	0.78
27	L	N	0.85
33	L	H	-0.47
33	L	N	-0.09
33	L	P	-1.23
33	L	V	-0.89
39	L	P	-1.53
24	N	S	0.22
26	P	G	-0.15
30	Q	W	-0.0
31	Q	A	-1.31
31	Q	S	-0.51
19	R	D	0.21
19	R	Q	0.12
19	R	T	-0.69
20	S	W	0.56
7	S	T	-0.06
7	S	W	0.13
7	S	Y	0.58
14	V	M	0.16
28	W	K	-1.02
28	W	S	-1.02
"""))



unirep_villin_predictions = np.load(
    "../../../data/predictions/rocklin_ssm2__villin_ssm2_stability__all_1900__test__predictions.npy"
)

assert unirep_villin_predictions.shape[0] == villin_res.shape[0]
assert unirep_villin_predictions.shape[0] == ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True].shape[0]

spearmanr(unirep_villin_predictions, ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True]['phenotype'])

spearmanr(villin_res.iloc[:,-1].values, ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True]['phenotype'])



from scipy.stats import ttest_ind_from_stats

ttest_ind_from_stats(mean1=spearmanr(unirep_villin_predictions, ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True]['phenotype'].values)[0], 
                     std1=compute_std(unirep_villin_predictions, ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True]['phenotype'].values), 
                     nobs1=30,
                     mean2=spearmanr(villin_res.iloc[:,-1].values, ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True]['phenotype'].values)[0], 
                     std2 =compute_std(villin_res.iloc[:,-1].values, ssm2[ssm2.phenotype_name.str.contains("villin") & ssm2.is_test == True]['phenotype'].values), 
                     nobs2=30)



get_wt_and_print_test_set_mutations('Pin1')

pin1_res = pd.read_table(StringIO("""
Position	Wild-type	mutant type	ddG
3	A	I	0.43
3	A	K	0.5
3	A	L	0.42
32	A	L	0.73
13	E	H	-0.03
5	E	D	0.28
6	E	L	0.07
6	E	W	0.29
26	F	L	-2.0
26	F	R	-0.16
11	G	H	-0.73
11	G	N	-1.05
11	G	T	-1.72
11	G	V	-2.0
21	G	L	-1.08
21	G	N	-0.53
21	G	T	-0.88
40	G	I	0.22
28	H	F	0.12
28	H	I	0.12
29	I	N	0.1
14	K	L	0.29
14	K	S	0.66
8	L	A	-1.19
8	L	H	-0.88
16	M	Q	0.35
16	M	Y	-0.11
2	M	D	0.92
2	M	P	0.19
2	M	R	0.82
27	N	D	-1.15
27	N	G	-3.41
27	N	M	-2.36
31	N	G	-0.76
31	N	Q	0.26
38	P	L	-2.79
38	P	S	-1.66
9	P	G	-0.83
9	P	R	-1.82
9	P	S	-1.11
10	P	R	0.67
10	P	W	0.8
15	R	H	0.39
15	R	L	-0.36
15	R	W	0.33
18	R	V	0.14
18	R	Y	0.2
22	R	S	-0.56
22	R	Y	0.01
37	R	I	-0.68
19	S	W	0.21
20	S	Q	0.52
33	S	H	-0.55
33	S	I	-1.1
33	S	P	-0.5
39	S	F	-1.02
39	S	R	-0.73
23	V	Q	-0.73
12	W	R	-2.08
35	W	I	-1.61
35	W	Q	-1.29
24	Y	S	-2.32
25	Y	S	-2.41
25	Y	T	-1.99
"""))


unirep_pin1_predictions = np.load(
    "../../../data/predictions/rocklin_ssm2__Pin1_ssm2_stability__all_1900__test__predictions.npy"
)




assert unirep_pin1_predictions.shape[0] == pin1_res.shape[0]
assert unirep_pin1_predictions.shape[0] == ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True].shape[0]



spearmanr(unirep_pin1_predictions, ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True]['phenotype'])



spearmanr(pin1_res.iloc[:,-1].values, ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True]['phenotype'])

from scipy.stats import ttest_ind_from_stats

ttest_ind_from_stats(mean1=spearmanr(unirep_pin1_predictions, ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True]['phenotype'].values)[0], 
                     std1=compute_std(unirep_pin1_predictions, ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True]['phenotype'].values), 
                     nobs1=30,
                     mean2=spearmanr(pin1_res.iloc[:,-1].values, ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True]['phenotype'].values)[0], 
                     std2 =compute_std(pin1_res.iloc[:,-1].values, ssm2[ssm2.phenotype_name.str.contains("Pin1") & ssm2.is_test == True]['phenotype'].values), 
                     nobs2=30)
                     

                     
