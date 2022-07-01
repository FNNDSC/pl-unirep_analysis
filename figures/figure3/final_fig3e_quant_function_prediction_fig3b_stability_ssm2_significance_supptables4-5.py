

import os
import pandas as pd
import sys
import os
from subprocess import call

#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
sys.path.append("../../")

from common_v2.validation_tools import regr_datasets, subsets, metrics, reps, transfer_datasets, pearson
import common_v2.validation_tools
from IPython.display import display, HTML
import numpy as np

import random
from scipy.stats import ttest_1samp
from sklearn.externals import joblib
from matplotlib.lines import Line2D

import common_v2.plot_style_utils as plot_style_utils
from common_v2.plot_style_utils import task_names

random.seed(42)
np.random.seed(42)



run_type = 'test' # change to 'validate' and rerun the notebook if interested in the validation set scores

"""
Data import and helpful functions putting the data in a nicer format
"""


#path_to_pieces = f"../data/pieces_new/"
data_path = "../../../data/"

stds = pd.read_csv(f"{data_path}std_results_val_resamp.csv", index_col=0)

regr_datasets.loc[9] = 'rocklin_ssm2_nat_eng'

def get_tab_dict(metric, to_plot=False):
    
    fig_n=1

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
                        f"{data_path}results/{d}__{s}__{rep}__{run_type}__regression_results.csv", header=None, index_col=0
                    ).loc[metric].iloc[0]
                    std_dict[rep] = stds.loc[f"{d}__{s}__{rep}__{run_type}"][metric]
                except Exception as e:
                    print(e)
            try:
                ser = pd.Series(rep_dict).sort_values(ascending=False)
                tab = pd.concat([ser, pd.Series(std_dict).loc[ser.index]], axis=1)
                tab.columns = ['avg','stdev']
                if to_plot:
                    
                    tab_to_plot = tab.copy()
                    
                    tab_to_plot.index = [plot_style_utils.rep_names[n] for n in tab_to_plot.index]
                    
                tab_dict[(d,s)] = tab
            except Exception as e:
                print(e)
    return tab_dict

all_df = pd.Series(get_tab_dict('mse'))
ascending = True
#our_rep = 'all_1900'

d = 'arnold_T50'
s = 'T50'

final_rep_rough_names = ['all_1900','best_other_rep', 'RGN', 'best_arnold_rep', ]

def get_pub_reps_only(single_subset_df, by='avg'):

    best_arnold_name = single_subset_df.loc[[
        "arnold_original_3_7", 
        "arnold_scrambled_3_5", 
        "arnold_random_3_7", 
        "arnold_uniform_4_1"
    ]].sort_values(by=by,ascending=ascending).iloc[0].name

    best_other_name = single_subset_df.loc[[
        "simple_freq_plus","simple_freq_and_len",
        "tfidf_3grams", "3grams", "tfidf_2grams", "2grams"
    ]].sort_values(by=by,ascending=ascending).iloc[0].name
    
    to_return = single_subset_df.loc[['all_1900',best_other_name,'RGN', best_arnold_name]] #.sort_values(by='avg',ascending=ascending)

    to_return.index = final_rep_rough_names
    
    return to_return



all_df_pub_reps = all_df.copy().map(get_pub_reps_only)

"""

Fig 3b: statistical significance for comparisons against baselines for individual proteins
"""
all_df_for_rocklin_fig = pd.Series(get_tab_dict('pearson_r'))

d = 'rocklin_ssm2'
s = 'villin_ssm2_stability'

ascending=True



from scipy.stats import ttest_ind_from_stats


size = 10000

b_reps = [
         "mean",
        "simple_freq_plus","simple_freq_and_len",
        "tfidf_3grams", "3grams", "tfidf_2grams", "2grams"] # "RGN", "arnold_original_3_7", "arnold_scrambled_3_5", "arnold_random_3_7", "arnold_uniform_4_1",

for s in subsets[d]:
    print(d,s)
        
    pvalue = ttest_ind_from_stats(
        mean1 = all_df_for_rocklin_fig[d][s].loc[['all_1900']].sort_values(by='avg',ascending=ascending).avg.iloc[0],
        std1 = all_df_for_rocklin_fig[d][s].loc[['all_1900']].sort_values(by='avg',ascending=ascending).stdev.iloc[0],
        nobs1=30,
        mean2 = all_df_for_rocklin_fig[d][s].loc[b_reps].sort_values(by='avg',ascending=ascending).avg.iloc[0],
        std2 = all_df_for_rocklin_fig[d][s].loc[b_reps].sort_values(by='avg',ascending=ascending).stdev.iloc[0],
        nobs2=30,
        equal_var = False
    )
    print(pvalue[0],pvalue[1])
    print("\n")



for s in all_df_pub_reps['rocklin_ssm2'].index:
    if s != 'full':
       
        all_df_pub_reps.drop(('rocklin_ssm2', s), inplace=True)

for d in ['solubility', 'leuenberger']:
    all_df_pub_reps.drop(d, inplace=True)
    
all_df_pub_reps.drop(('fowler','full'), inplace=True)



final_table = pd.DataFrame(index=final_rep_rough_names, columns= all_df_pub_reps.index)


for d,s in final_table.columns:
    for rep in final_table.index:
        final_table.loc[rep, (d,s)] = all_df_pub_reps[d][s].loc[rep].avg


final_table_std = pd.DataFrame(index=final_rep_rough_names, columns= all_df_pub_reps.index)

for d,s in final_table_std.columns:
    for rep in final_table_std.index:
        final_table_std.loc[rep, (d,s)] = all_df_pub_reps[d][s].loc[rep].stdev
        


def highlight_min(s):
    '''
    highlight the maximum in a Series green.
    '''
    is_min = s == s.min()
    return ['background-color: lightgreen' if v else '' for v in is_min]

def highlight_max(s):
    '''
    highlight the maximum in a Series green.
    '''
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]


final_table.style.apply(highlight_min, axis=0).set_precision(3)


from scipy.stats import ttest_ind_from_stats

size = 10000

for d,s in final_table_std.columns[:4]:
    print(d,s)
    
    pvalue = ttest_ind_from_stats(
        mean1 = final_table[d][s][['all_1900',]].sort_values(ascending=ascending).iloc[0],
        std1 = final_table_std[d][s][['all_1900']].sort_values(ascending=ascending).iloc[0],
        nobs1=30,
        mean2 = final_table[d][s][['best_other_rep','best_arnold_rep', 'RGN']].sort_values(ascending=ascending).iloc[0],
        std2 = final_table_std[d][s][['best_other_rep','best_arnold_rep', 'RGN']].sort_values(ascending=ascending).iloc[0],
        nobs2=30,
        equal_var = False
    )
    print(pvalue)
    


from scipy.stats import ttest_ind_from_stats

size = 10000

for d,s in final_table_std.columns[:4]:
    print(d,s)
    
    pvalue = ttest_ind_from_stats(
        mean1 = final_table[d][s][['best_other_rep',]].sort_values(ascending=ascending).iloc[0],
        std1 = final_table_std[d][s][['best_other_rep']].sort_values(ascending=ascending).iloc[0],
        nobs1=30,
        mean2 = final_table[d][s][['all_1900','best_arnold_rep', 'RGN']].sort_values(ascending=ascending).iloc[0],
        std2 = final_table_std[d][s][['all_1900','best_arnold_rep', 'RGN']].sort_values(ascending=ascending).iloc[0],
        nobs2=30,
        equal_var = False
    )
    print(pvalue)



from scipy.stats import ttest_ind_from_stats

size = 10000

for d,s in final_table_std.columns[:4]:
    print(d,s)
    
    pvalue = ttest_ind_from_stats(
        mean1 = final_table[d][s][['best_other_rep',]].sort_values(ascending=ascending).iloc[0],
        std1 = final_table_std[d][s][['best_other_rep']].sort_values(ascending=ascending).iloc[0],
        nobs1=30,
        mean2 = final_table[d][s][['best_arnold_rep']].sort_values(ascending=ascending).iloc[0],
        std2 = final_table_std[d][s][['best_arnold_rep']].sort_values(ascending=ascending).iloc[0],
        nobs2=30,
        equal_var = False
    )
    print(pvalue)
    
size = 10000

for d,s in final_table_std.columns[4:]:
    print(d,s)
    
    pvalue = ttest_ind_from_stats(
        mean1 = final_table[d][s][['all_1900',]].sort_values(ascending=ascending).iloc[0],
        std1 = final_table_std[d][s][['all_1900']].sort_values(ascending=ascending).iloc[0],
        nobs1=30,
        mean2 = final_table[d][s][['best_other_rep','best_arnold_rep', 'RGN']].sort_values(ascending=ascending).iloc[0],
        std2 = final_table_std[d][s][['best_other_rep','best_arnold_rep', 'RGN']].sort_values(ascending=ascending).iloc[0],
        nobs2=30,
        equal_var = False
    )
    print(pvalue)

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:75% !important; }</style>"))


intermediate = final_table.loc[:,['fowler']].copy()

intermediate.index = [plot_style_utils.main_text_rep_names[n] for n in intermediate.index]

#intermediate.columns = [task_names[n] for n in intermediate.columns.levels[1][intermediate.columns.labels[1]]]
intermediate.columns = ['TEM1','E1 Ubiquitin','Gb1','HSP90','Kka2','Pab1','PSD95pdz3','Ubiquitin','Yap65']

functions = ['Hydrolysis','E1 \nactivation','IgG-binding','Chaperone',
             'Kinase\nactivity', 'Poly-A\nbinding', 'Kinase\nbinding','Partner\nbinding', 'Binding\nsubstrate']

intermediate.style.apply(highlight_min, axis=0).set_precision(3).set_table_styles(
        [
            dict(selector="th",props=[('max-width', '100px'), ('text-align','center')]),
            dict(selector="td",props=[('text-align','center')])
        ]
    )



palette = plot_style_utils.set_pub_plot_context(colors='categorical', context="poster")




fig = plt.figure(figsize=(4.74*2, 2*2))
ax = fig.subplots()
intermediate.T.plot(marker='o',linewidth=0, ax=ax, legend=None)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(range(9),intermediate.columns, rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.ylim(0,0.148)
#plt.xticks(range(9),functions, rotation=0)

#plt.title("Mean Squared Error - Function Prediction")

#plot_style_utils.save_for_pub(fig=fig, path="./figures/functional_datasets_names", dpi=250)



fig = plt.figure(figsize=(4.74*2, 2*2))
ax = fig.subplots()
intermediate.T.plot(marker='o',linewidth=0, ax=ax, legend=None)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.xticks(range(9),intermediate.columns, rotation=0, fontsize=12)
plt.xticks(range(9),functions, rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0,0.148)

#plt.title("Mean Squared Error - Function Prediction")

plot_style_utils.save_for_pub(fig=fig, path="./functional_datasets", dpi=250)

"""
Fig 3b: statistical significance for comparisons against baselines for individual proteins
"""
fowler_seqs = pd.Series({'Kka2': 'MIEQDGLHAGSPAAWVERLFGYDWAQQTIGCSDAAVFRLSAQGRPVLFVKTDLSGALNELQDEAARLSWLATTGVPCAAVLDVVTEAGRDWLLLGEVPGQDLLSSHLAPAEKVSIMADAMRRLHTLDPATCPFDHQAKHRIERARTRMEAGLVDQDDLDEEHQGLAPAELFARLKARMPDGEDLVVTHGDACLPNIMVENGRFSGFIDCGRLGVADRYQDIALATRDIAEELGGEWADRFLVLYGIAAPDSQRIAFYRLLDEFF',
 'PSD95pdz3': 'MDCLCIVTTKKYRYQDEDTPPLEHSPAHLPNQANSPPVIVNTDTLEAPGYELQVNGTEGEMEYEEITLERGNSGLGFSIAGGTDNPHIGDDPSIFITKIIPGGAAAQDGRLRVNDSILFVNEVDVREVTHSAAVEALKEAGSIVRLYVMRRKPPAEKVMEIKLIKGPKGLGFSIAGGVGNQHIPGDNSIYVTKIIEGGAAHKDGRLQIGDKILAVNSVGLEDVMHEDAVAALKNTYDVVYLKVAKPSNAYLSDSYAPPDITTSYSQHLDNEISHSSYLGTDYPTAMTPTSPRRYSPVAKDLLGEEDIPREPRRIVIHRGSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQTVTIIAQYKPEEYSRFEAKIHDLREQLMNSSLGSGTASLRSNPKRGFYIRALFDYDKTKDCGFLSQALSFRFGDVLHVIDAGDEEWWQARRVHSDSETDDIGFIPSKRRVERREWSRLKAKDWGSSSGSQGREDSVLSYETVTQMEVHYARPIIILGPTKDRANDDLLSEFPDKFGSCVPHTTRPKREYEIDGRDYHFVSSREKMEKDIQAHKFIEAGQYNSHLYGTSVQSVREVAEQGKHCILDVSANAVRRLQAAHLHPIAIFIRPRSLENVLEINKRITEEQARKAFDRATKLEQEFTECFSAIVEGDSFEEIYHKVKRVIEDLSGPYIWVPARERL',
 'Pab1': 'MADITDKTAEQLENLNIQDDQKQAATGSESQSVENSSASLYVGDLEPSVSEAHLYDIFSPIGSVSSIRVCRDAITKTSLGYAYVNFNDHEAGRKAIEQLNYTPIKGRLCRIMWSQRDPSLRKKGSGNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAPHLSRKERDSQLEETKAHYTNLYVKNINSETTDEQFQELFAKFGPIVSASLEKDADGKLKGFGFVNYEKHEDAVKAVEALNDSELNGEKLYVGRAQKKNERMHVLKKQYEAYRLEKMAKYQGVNLFVKNLDDSVDDEKLEEEFAPYGTITSAKVMRTENGKSKGFGFVCFSTPEEATKAITEKNQQIVAGKPLYVAIAQRKDVRRSQLAQQIQARNQMRYQQATAAAAAAAAGMPGQFMPPMFYGVMPPRGVPFNGPNPQQMNPMGGMPKNGMPPQFRNGPVYGVPPQGGFPRNANDNNQFYQQKQRQALGEQLYKKVSAKTSNEEAAGKITGMILDLPPQEVFPLLESDELFEQHYKEASAAYESFKKEQEQQTEQA',
 'TEM-1': 'MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW',
 'UBI4': 'MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGMQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGN',
 'Yap65': 'MDPGQQPPPQPAPQGQGQPPSQPPQGQGPPSGPGQPAPAATQAAPQAPPAGHQIVHVRGDSETDLEALFNAVMNPKTANVPQTVPMRLRKLPDSFFKPPEPKSHSRQASTDAGTAGALTPQHVRAHSSPASLQLGAVSPGTLTPTGVVSGPAATPTAQHLRQSSFEIPDDVPLPAGWEMAKTSSGQRYFLNHIDQTTTWQDPRKAMLSQMNVTAPTSPPVQQNMMNSASGPLPDGWEQAMTQDGEIYYINHKNKTTSWLDPRLDPRFAMNQRISQSAPVKQPPPLAPQSPQGGVMGGSNSNQQQQMRLQQLQMEKERLRLKQQELLRQAMRNINPSTANSPKCQELALRSQLPTLEQDGGTQNPVSSPGMSQELRTMTTNSSDPFLNSGTYHSRDESTDSGLSMSSYSVPRTPDDFLNSVDEMDTGDTINQSTLPSQQNRFPDYLEAIPGTNVDLGTLEGDGMNIEGEELMPSLQEALSSDILNDMESVLAATKLDKESFLTWL',
 'gb1': 'MEKEKKVKYFLRKSAFGLASVSAAFLVGSTVFAVDSPIEDTPIIRNGGELTNLLGNSETTLALRNEESATADLTAAAVADTVAAAAAENAGAAAWEAAAAADALAKAKADALKEFNKYGVSDYYKNLINNAKTVEGIKDLQAQVVESAKKARISEATDGLSDFLKSQTPAEDTVKSIELAEAKVLANRELDKYGVSDYHKNLINNAKTVEGVKELIDEILAALPKTDQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTEKPEVIDASELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPGDAPTEPEKPEASIPLVPLTPATPIAKDDAKKDDTKKEDAKKPEAKKDDAKKAETLPTTGEGSNPFFTAAALAVMAGAGALAVASKRKED',
 'hsp90': 'MASETFEFQAEITQLMSLIINTVYSNKEIFLRELISNASDALDKIRYKSLSDPKQLETEPDLFIRITPKPEQKVLEIRDSGIGMTKAELINNLGTIAKSGTKAFMEALSAGADVSMIGQFGVGFYSLFLVADRVQVISKSNDDEQYIWESNAGGSFTVTLDEVNERIGRGTILRLFLKDDQLEYLEEKRIKEVIKRHSEFVAYPIQLVVTKEVEKEVPIPEEEKKDEEKKDEEKKDEDDKKPKLEEVDEEEEKKPKTKKVKEEVQEIEELNKTKPLWTRNPSDITQEEYNAFYKSISNDWEDPLYVKHFSVEGQLEFRAILFIPKRAPFDLFESKKKKNNIKLYVRRVFITDEAEDLIPEWLSFVKGVVDSEDLPLNLSREMLQQNKIMKVIRKNIVKKLIEAFNEIAEDSEQFEKFYSAFSKNIKLGVHEDTQNRAALAKLLRYNSTKSVDELTSLTDYVTRMPEHQKNIYYITGESLKAVEKSPFLDALKAKNFEVLFLTDPIDEYAFTQLKEFEGKTLVDITKDFELEETDEEKAEREKEIKEYEPLTKALKEILGDQVEKVVVSYKLLDAPAAIRTGQFGWSANMERIMKAQALRDSSMSSYMSSKKTFEISPKSPIIKELKKRVDEGGAQDKTVKDLTKLLYETALLTSGFSLDEPTSFASRINRLISLGLNIDEDEETETAPEASTAAPVEEVPADTEMEEVD'})
 
 fowler_seqs.map(len).sort_values()


import Levenshtein



distances = pd.DataFrame(index = fowler_seqs.index, columns = fowler_seqs.index)
for i in fowler_seqs.index:
    for j in fowler_seqs.index:
        distances.loc[i,j] = Levenshtein.ratio(fowler_seqs.loc[i],fowler_seqs.loc[j])




def mut(st):
    sub_position = np.random.randint(0,len(st))
    return st[:sub_position] + 'A' + st[sub_position+1:]
fowler_seqs.map(lambda x: np.mean([Levenshtein.ratio(x, mut(x)) for i in range(100)])).mean()


"""average similarity between 8 proteins in this analysis:"""


distances.mean().mean()
"""
How much better are we at quantitative function prediction?"""

print(intermediate.loc['Best Doc2Vec'] / intermediate.loc['UniRep Fusion'])
print('mean',(intermediate.loc['Best Doc2Vec'] / intermediate.loc['UniRep Fusion']).mean())

"""

The non Quantitative function prediction parts of Supp. Table S4-S5:
Small scale function prediction tasks from Yang 2018
"""


intermediate = final_table.loc[:,['arnold_T50', 'arnold_absorption', 'arnold_enantioselectivity',
       'arnold_localization']].copy()
intermediate.index = [plot_style_utils.main_text_rep_names[n] for n in intermediate.index]

intermediate.columns = [task_names[n] for n in intermediate.columns.levels[0][intermediate.columns.labels[0]]]

#intermediate.columns = [f"plot_style_utils." d,s in intermediate.columns]

intermediate.style.apply(
    highlight_min, axis=0
).set_precision(3).set_table_styles(
        [
            dict(selector="th",props=[('max-width', '120px'), ('text-align','center')]),
            dict(selector="td",props=[('text-align','center')])
        ]
    )



intermediate = final_table.loc[:,['rocklin_ssm2','rocklin_all_rds']].copy()

intermediate.index = [plot_style_utils.main_text_rep_names[n] for n in intermediate.index]

intermediate.columns = [task_names[n] for n in intermediate.columns.levels[0][intermediate.columns.labels[0]]]

#.style.apply(highlight_min, axis=0).set_precision(3)
intermediate.style.apply(highlight_min, axis=0).set_precision(3).set_table_styles(
        [
            dict(selector="th",props=[('max-width', '120px'), ('text-align','center')]),
            dict(selector="td",props=[('text-align','center')])
        ]
    )



display(intermediate.loc['Best Doc2Vec'] / intermediate.loc['UniRep Fusion'])
print('mean',(intermediate.loc['Best Doc2Vec'] / intermediate.loc['UniRep Fusion']).mean())



# Holds in separate analysis for natural and engineered proteins too!

intermediate = final_table.loc[:,['rocklin_ssm2_nat_eng']].copy()

intermediate.index = [plot_style_utils.main_text_rep_names[n] for n in intermediate.index]

intermediate.columns = [task_names[n] for n in intermediate.columns.levels[1][intermediate.columns.labels[1]]]

#.style.apply(highlight_min, axis=0).set_precision(3)
intermediate.style.apply(highlight_min, axis=0).set_precision(3).set_table_styles(
        [
            dict(selector="th",props=[('max-width', '120px'), ('text-align','center')]),
            dict(selector="td",props=[('text-align','center')])
        ]
    )


