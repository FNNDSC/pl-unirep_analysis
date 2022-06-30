# This is a cell to hide code snippets from displaying
# This must be at first cell!

from IPython.display import HTML

hide_me = ''
HTML('''<script>
code_show=true; 
function code_toggle() {
  if (code_show) {
    $('div.input').each(function(id) {
      el = $(this).find('.cm-variable:first');
      if (id == 0 || el.text() == 'hide_me') {
        $(this).hide();
      }
    });
    $('div.output_prompt').css('opacity', 0);
  } else {
    $('div.input').each(function(id) {
      $(this).show();
    });
    $('div.output_prompt').css('opacity', 1);
  }
  code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input style="opacity:0" type="submit" value="Click here to toggle on/off the raw code."></form>''')
from IPython.core.display import HTML
HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""")
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

sys.path.append('../../')


from common_v2.plot_style_utils import task_names
import common_v2.plot_style_utils
import common_v2.plot_style_utils as plot_style_utils
from common_v2.validation_tools import regr_datasets, subsets, metrics, reps, transfer_datasets, pearson
import common_v2.validation_tools

random.seed(42)
np.random.seed(42)



path_to_pieces = f"../../../data/pieces_new/"

stds = pd.read_csv("../../../data/std_results_val_resamp.csv", index_col=0)

regr_datasets.loc[9] = 'rocklin_ssm2_nat_eng'


common_v2.plot_style_utils.set_pub_plot_context(colors='categorical', context="poster")

palette = sns.color_palette()



uni_color, rgn_color, arnold_color, baseline_color, fusion_color = [0,1,2,3,5]

rep_colors ={
    'RGN': palette[rgn_color],
    '64_avg_hidden': palette[uni_color],
    '64_final_hidden': palette[uni_color],
    '64_final_cell': palette[uni_color],
    '256_avg_hidden': palette[uni_color],
    '256_final_cell': palette[uni_color],
    'avg_hidden': palette[uni_color],
    'final_hidden': palette[uni_color],
    'final_cell': palette[uni_color],
    'arnold_original_3_7': palette[arnold_color],
    'arnold_scrambled_3_5': palette[arnold_color],
    'arnold_random_3_7': palette[arnold_color],
    'arnold_uniform_4_1': palette[arnold_color],
    'all_64': palette[uni_color],
    'all_256': palette[uni_color],
    'all_1900': palette[uni_color],
    'all_avg_hidden': palette[uni_color],
    'all_final_cell': palette[uni_color],
    'RGN_avg_hidden': palette[fusion_color],
    'RGN_final_cell': palette[fusion_color],
    'simple_freq_plus': palette[baseline_color],
    'simple_freq_and_len': palette[baseline_color],
    '2grams':  palette[baseline_color],
    '3grams':  palette[baseline_color],
    'tfidf_2grams': palette[baseline_color],
    'tfidf_3grams': palette[baseline_color],
    'mean': palette[baseline_color]
}

group_colors = {
    "UniRep": palette[uni_color], 
    "RGN": palette[rgn_color], 
    "Doc2Vec": palette[arnold_color], 
    "Baseline": palette[baseline_color], 
    "Fusion": palette[fusion_color]
}

scratch = pd.Series({
    "RGN": 0.5, 
    "Doc2Vec": 0.5, 
    "Baseline": 0.5, 
    "Fusion": 0.5,
    "UniRep": 0.5
})

l_fig, ax = plt.subplots(1, 1)

scratch.plot(kind='barh', color=[group_colors[n] for n in scratch.index], xticks=[], grid=0, figsize=(.7,1.3), ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.close()

#plot_style_utils.save_for_pub(l_fig, path=f"./figures/legend", dpi=500)
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
    
  

regr_datasets = regr_datasets.drop([0,1])

d = 'leuenberger'
s = 'ecoli_tm'
rep = 'avg_hidden'
run_type = 'test'
metric = 'pearson_r'

to_plot = False

fig_n = 1

def get_tab_dict_and_plot(metric, to_plot=False, figsize=(7,7)):
    
    global fig_n
    
    tab_dict = {}

    for d in regr_datasets:
        ss = subsets[d]
        if (len(ss) > 1) & (d != 'rocklin_ssm2_nat_eng'):
            ss = list(ss)+["full"]
        for s in ss:
            rep_dict = {}
            std_dict = {}
            for rep in reps:
                try:
                    rep_dict[rep] = pd.read_csv(
                        f"../../../data/results/{d}__{s}__{rep}__{run_type}__regression_results.csv", header=None, index_col=0
                    ).loc[metric].iloc[0]
                    std_dict[rep] = stds.loc[f"{d}__{s}__{rep}__{run_type}"][metric]
                except Exception as e:
                    print(e)
                    #print(f"could not find ./results/{d}__{s}__{rep}__{run_type}__regression_results.csv")
            #print(f"{d} {s}")
            try:
                ser = pd.Series(rep_dict).sort_values(ascending=False)
                tab = pd.concat([ser, pd.Series(std_dict).loc[ser.index]], axis=1)
                tab.columns = ['avg','stdev']
                if to_plot:
                    
                    tab_to_plot = tab.copy()
                    
                    tab_to_plot.index = [common_v2.plot_style_utils.rep_names[n] for n in tab_to_plot.index]
                    
                    #display(tab)
                    
                    fig, ax = plt.subplots(1, 1)
                    tab_to_plot.avg.plot(kind='barh', color=[rep_colors[n] for n in tab.index], figsize=figsize, 
                            xerr=tab_to_plot.stdev, ax=ax
                        )
                    plt.xlabel("Mean Squared Error")
                    #display(l_fig)
                    display(fig)
                    plt.close()
                    if len(subsets[d]) > 1:
                        if s != 'full':
                            printmd(f"Supplementary Results Figure {fig_n}: All representation results in {task_names[d].lower()} task. Subset: {task_names[s]}. \n")
                        else:
                            printmd(f"Supplementary Results Figure {fig_n}: All representation results in {task_names[d].lower()} task, {task_names[s]}. \n")
                    else:
                        printmd(f"Supplementary Results Figure {fig_n}: All representation results in {task_names[d].lower()} task. \n")
                    
                    #custom_lines = [Line2D([0], [0], color=rep_colors[o], lw=4) for o in tab.index.tolist()]
                    #ax.legend(custom_lines, tab.index.tolist())

                    #plot_style_utils.save_for_pub(fig, path=f"./figures/{fig_n}__{d}__{s}__{run_type}", dpi=500)
                    #print('saving',f"./figures/{d}__{s}__{run_type}")
                    fig_n+=1
                    #plt.title(f"{d}__{s}__{run_type}")
                tab_dict[(d,s)] = tab
            except Exception as e:
                print(e)
    return tab_dict  



# 9*6 inches

tab_dict = get_tab_dict_and_plot('mse', to_plot=True, figsize=(6,7))


tab_dict_nontransfer = tab_dict.copy()



tab_dict_nontransfer



regr_datasets = ['leuenberger', 'solubility']

tab_dict = get_tab_dict_and_plot('mse', to_plot=True, figsize=(6,7))
tab_dict_excluded = tab_dict.copy()
metric = 'transfer_ratio_avg'
ascending = False

def get_transfer_result_dfs(path_to_results_folder, prefix="transfer__"):
    li = pd.Series(os.listdir(path_to_results_folder))
    li = li[li.str.contains("transfer") & li.str.contains("results") & li.str.contains(run_type)]
    
    res_dfs = {}
        
    for dataset_name in transfer_datasets:
        
        df = pd.concat([
            pd.read_csv(
                        os.path.join(path_to_results_folder,line),
                        index_col=0
                    ) for line in li[li.str.startswith(prefix+dataset_name+"__")]
        ])
        df['perf_ratio_avg'] = df['transfer_ratio_avg'] / df['indomain_ratio_avg']
        
        res_dfs[dataset_name] = df
    
    return res_dfs

# This cell computes std dataframe from errors in the results folder - this is std of results 
# from different hold-one-out runs. Has the same index and columns as the original results dataframes in the 
# dict above - this is for passing in xerr argument in plot
def get_transfer_std_dfs(path_to_results_folder, prefix="transfer__"):

    li = pd.Series(os.listdir(path_to_results_folder))
    li = li[li.str.contains("transfer") & li.str.contains("metrics")  & li.str.contains(run_type)]

    std_dfs = {}
    for d in transfer_datasets:
        std_df = pd.DataFrame(index=reps, columns=['transfer_ratio_avg', 'indomain_ratio_avg', 'perf_ratio_avg'])
        for rep in reps:
            df = pd.read_csv(os.path.join(path_to_results_folder, 
                                                       li[
                                                             li.str.startswith(prefix+d+"__") & li.str.contains(rep)
                                                        ].iloc[0]),
                                          index_col=0)
            df['perf_ratio'] = df['transfer_ratio'] / df['indomain_ratio']
            std_df.loc[rep] = df[['transfer_ratio', 'indomain_ratio', 'perf_ratio']].std().values
        std_dfs[d] = std_df

    return std_dfs

def get_original_dfs(path_to_results_folder, prefix="transfer__"):

    li = pd.Series(os.listdir(path_to_results_folder))
    li = li[li.str.contains("transfer") & li.str.contains("metrics")  & li.str.contains(run_type)]

    original_dfs = {}
    for d in transfer_datasets:
        o_df = {}
        for rep in reps:
            df = pd.read_csv(os.path.join(path_to_results_folder, 
                                                       li[
                                                             li.str.startswith(prefix+d+"__") & li.str.contains(rep)
                                                        ].iloc[0]),
                                          index_col=0)
            o_df[rep] = df
        original_dfs[d] = o_df
    return original_dfs

path_to_results_folder = "../../../data/results/"
transfer_results = get_transfer_result_dfs(path_to_results_folder)
std_dfs = get_transfer_std_dfs(path_to_results_folder)
original_dfs = get_original_dfs(path_to_results_folder)

group_assignment = {'RGN': 'rgn',
 '64_avg_hidden': 'uni',
 '64_final_hidden': 'uni',
 '64_final_cell': 'uni',
 '256_avg_hidden': 'uni',
 '256_final_cell': 'uni',
 'avg_hidden': 'uni',
 'final_hidden': 'uni',
 'final_cell': 'uni',
 'arnold_original_3_7': 'arnold',
 'arnold_scrambled_3_5': 'arnold',
 'arnold_random_3_7': 'arnold',
 'arnold_uniform_4_1': 'arnold',
 'all_64': 'uni',
 'all_256': 'uni',
 'all_1900': 'uni',
 'all_avg_hidden': 'uni',
 'all_final_cell': 'uni',
 'RGN_avg_hidden': 'fusion',
 'RGN_final_cell': 'fusion',
 'simple_freq_plus': 'baseline',
 'simple_freq_and_len': 'baseline',
 '2grams': 'baseline',
 '3grams': 'baseline',
 'tfidf_2grams': 'baseline',
 'tfidf_3grams': 'baseline',
 'mean': 'baseline'}

marker_assignment = {
                     
 'all_1900': 'o',
 'all_64': 'D',
 'all_256': '*',
 'all_avg_hidden': 'p',
 'all_final_cell': '<',
 'avg_hidden': 'h',
 'final_hidden': '^',
 'final_cell': 'X',
                     
 '64_avg_hidden': 's',
 '64_final_hidden': '>',
 '64_final_cell': '$\mathbf{<}$',
 '256_avg_hidden': '$V$',
 '256_final_cell': '$L$',

 
 'RGN': '$\mathbf{o}$',
                     
 'arnold_original_3_7': '^',
 'arnold_scrambled_3_5': 'v',
 'arnold_random_3_7': '>',
 'arnold_uniform_4_1': '<',

 'RGN_avg_hidden': 'X',
 'RGN_final_cell': 'd',
                     
 'simple_freq_plus': '$\mathbf{+}$',
 'simple_freq_and_len': '$\mathbf{f}$',
 '2grams': 'D',
 '3grams': '$\mathbf{V}$',
 'tfidf_2grams': 'o',
 'tfidf_3grams': '$\mathbf{+}$',
 'mean': '$\mathbf{m}$'}

group_assignment_rev = pd.Series(group_assignment).to_frame()
group_assignment_rev = group_assignment_rev.reset_index().set_index(0)
group_assignment_rev.columns = ['rep']

def plot_2d_transfer(df, figsize=(6,8), horizalign=['left','right'], lims=False):
    df = df.loc[df.sort_values(by='transfer_ratio_avg').index]
    fig, ax = plt.subplots(figsize=figsize)
    for rep in df['indomain_ratio_avg'].index:
        #print(rep)
        color = rep_colors[rep]
        marker = marker_assignment[rep]# markers[group_assignment_rev.loc[group_assignment[rep]].reset_index()['rep'].reset_index().set_index('rep').loc[rep].values[0]]

        #print(rep,marker)
        ax.plot(df['indomain_ratio_avg'].loc[rep], df['transfer_ratio_avg'].loc[rep], 
                color=color, 
                marker= marker,
                mew = 0.25,
                mec='black',
                label= plot_style_utils.rep_names[rep])
    ax.legend(handlelength=0, fontsize='x-small', ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.text(df.loc['all_1900','indomain_ratio_avg'], df.loc['all_1900','transfer_ratio_avg'], plot_style_utils.rep_names['all_1900'], horizontalalignment=np.random.choice(horizalign), size='12', color='black') # , weight='semibold'

    if lims:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    plt.xlabel('Average In-Domain Ratio')
    plt.ylabel('Average Transfer Ratio')
    
#     for line in range(0,df.shape[0]):
#         if lims==False:
#             plt.text(df['indomain_ratio_avg'][line], df['transfer_ratio_avg'][line], df.index[line], horizontalalignment=np.random.choice(horizalign), size='12', color='black') # , weight='semibold'
#         else:
#             if (df['indomain_ratio_avg'][line] < lims[0][1]) & (df['transfer_ratio_avg'][line] < lims[1][1]) & (df['transfer_ratio_avg'][line] > lims[1][0]) & (df['indomain_ratio_avg'][line] > lims[0][0]):
#                 plt.text(df['indomain_ratio_avg'][line], df['transfer_ratio_avg'][line], df.index[line], horizontalalignment=np.random.choice(horizalign), size='12', color='black') # , weight='semibold'
    return fig, ax


transfer_results['rocklin_ssm2'].loc['all_1900']

transfer_datasets = ['fowler_consistent_single_UBI', 'rocklin_ssm2']


fig_n=44


run_type


for d in transfer_datasets:
    fig, ax = plot_2d_transfer(transfer_results[d], horizalign=['left'])
    display(fig)
    printmd(f"\n Supplementary Figure {fig_n}: All representation results in {task_names[d].lower()} transfer task. \n")
    #printmd("\n\n")
    plt.close()
    #common_v2.plot_style_utils.save_for_pub(fig=fig, path=f"./figures/transfer_{d}", dpi=500)
    
def get_spec_transfer_nonavg_result_dfs(d='rocklin_ssm2_nat_eng', withheld_subset = 'natural'):

    res_df = {}
    for rep in reps:

        df = pd.read_csv(f"../../../data/results/transfer__{d}__{rep}__{run_type}__metrics.csv", index_col=0).loc[withheld_subset]

        res_df[rep] = df
    
    res_df = pd.DataFrame(res_df).T
    
    res_df['perf_ratio'] = res_df['transfer_ratio'] / res_df['indomain_ratio']
    
    res_df = res_df[['transfer_ratio', 'indomain_ratio', 'perf_ratio']]
    res_df.columns = ['transfer_ratio_avg', 'indomain_ratio_avg', 'perf_ratio_avg'] # not actually avg, just easier
    
    return res_df   fig_n +=1




printmd(f"Supplementary Figure {fig_n}: \n All representation results in {task_names['rocklin_ssm2_remote_test'].lower()} transfer task. \n")
fig, ax = plot_2d_transfer(
    get_spec_transfer_nonavg_result_dfs(d='rocklin_ssm2_remote_test', withheld_subset = 'remote')
)
plt.xlabel("In-Domain Ratio")
plt.ylabel("Transfer Ratio")
display(fig)
plt.close()
#plot_style_utils.save_for_pub(fig=fig, path=f"./figures/transfer_rocklin_ssm2_remote_test", dpi=500)



writer = pd.ExcelWriter(f'./all_results_{run_type}.xlsx')


tab_dict_nontransfer

transfer_results['rocklin_ssm2_remote_test'] = get_spec_transfer_nonavg_result_dfs(d='rocklin_ssm2_remote_test', withheld_subset = 'remote')
transfer_results


for n in tab_dict_nontransfer.keys():
    sheet_name = plot_style_utils.task_names[n[0]]+" "+n[1]
    tab_dict_nontransfer[n].to_excel(writer, sheet_name=sheet_name)
    
for n in tab_dict_excluded.keys():
    sheet_name = plot_style_utils.task_names[n[0]]+" "+n[1]
    tab_dict_excluded[n].to_excel(writer, sheet_name=sheet_name)
    
for n in transfer_results.keys():
    sheet_name = plot_style_utils.task_names[n]
    transfer_results[n].to_excel(writer, sheet_name=sheet_name)




wildtypes['pearson_nn'].values


def one_hot_params(l):
    return 20*l+1

def UniRep_params(l):
    return 1900+1

max_l=1000

fig = plt.figure()
plt.plot(np.arange(max_l),[one_hot_params(l) for l in np.arange(max_l)], label='One-hot encoding')
plt.plot(np.arange(max_l),[UniRep_params(l) for l in np.arange(max_l)], label='UniRep')
plt.legend()
plt.xlabel("Sequence Length")
plt.ylabel("Number of\nmodel parameters")
plt.title("Complexity of UniRep vs one-hot protein encoding\n")
plt.xlim(0,1000)
plt.ylim(0,20000)
#plot_style_utils.save_for_pub(fig=fig, path="./figures/supplemental_complexity_unirep_one_hot")


fig, axs = plt.subplots(2,1)

plt.subplots_adjust(hspace = 0.5)

pd.Series({"UniRep":3,
          "Next best method (Doc2Vec)":300}).plot(kind='barh', ax=axs[0], xlim=(0,550), title="Recall Task")

pd.Series({"UniRep":3,
          "Next best method (Doc2Vec)":540}).plot(kind='barh', ax=axs[1], xlim=(0,550), sharex=axs[0], title="\nMaximum Brightness Task")
axs[1].set_xlabel("$ Thousands")

#plot_style_utils.save_for_pub(fig=fig, path="./figures/cost_savings", dpi=350)
