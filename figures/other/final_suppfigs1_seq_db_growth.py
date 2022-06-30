import pandas as pd
import io
%pylab inline
import sys
sys.path.append('../../')

import common_v2.plot_style_utils as plot_style_utils
# from http://www.rcsb.org/pdb/statistics/contentGrowthChart.do?content=total, retrieved on Tue Dec 4 2018

pdb_searcheable_records = pd.read_table(io.StringIO(
"""Year	Yearly	Total
2018	10491.0	146861.0
2017	11108.0	136370.0
2016	10837.0	125262.0
2015	9272.0	114425.0
2014	9578.0	105153.0
2013	9351.0	95575.0
2012	8765.0	86224.0
2011	7936.0	77459.0
2010	7752.0	69523.0
2009	7292.0	61771.0
2008	6906.0	54479.0
2007	7136.0	47573.0
2006	6409.0	40437.0
2005	5334.0	34028.0
2004	5149.0	28694.0
2003	4147.0	23545.0
2002	2994.0	19398.0
2001	2814.0	16404.0
2000	2627.0	13590.0
1999	2356.0	10963.0
1998	2057.0	8607.0
1997	1565.0	6550.0
1996	1173.0	4985.0
1995	941.0	3812.0
1994	1289.0	2871.0
1993	696.0	1582.0
1992	192.0	886.0
1991	187.0	694.0
1990	142.0	507.0
1989	74.0	365.0
1988	53.0	291.0
1987	25.0	238.0
1986	18.0	213.0
1985	20.0	195.0
1984	22.0	175.0
1983	36.0	153.0
1982	32.0	117.0
1981	16.0	85.0
1980	16.0	69.0
1979	11.0	53.0
1978	6.0	42.0
1977	23.0	36.0
1976	13.0	13.0
1975	0.0	0.0
1974	0.0	0.0
1973	0.0	0.0
1972	0.0	0.0
"""
))


pdb_searcheable_records.columns = ['Year', 'Yearly', 'PDB Structures']

plot_style_utils.set_pub_plot_context()
fig = plt.figure()
ax = fig.subplots()

pdb_searcheable_records[['Year','PDB Structures']].set_index("Year").loc[:2011].plot(ax=ax)

# from ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2016_10/relnotes.txt
#1st months
pd.Series({
    
    2011:3653743,
    2012:4606913,
    2013:6412887,
    2014:9370012,
    2015:11992242,
    2016:16038089,
    2017:20083468,
    2018:30071646
}).plot(label='UniRef50', ax=ax)

pd.Series({
    
    2011:24828830,
    2012:30362438,
    2013:36188164,
    2014:56323070,
    2015:79754489,
    2016:111569591,
    2017:143389247,
    2018:200779506
}).plot(label='UniParc', ax=ax)

plt.legend()
plt.ylabel("Log number of protein sequences")

#plot_style_utils.save_for_pub(fig=fig,path="./figures/sequence_db_growth")

