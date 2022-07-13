#
# unirep_analysis ds ChRIS plugin app
#
# (c) 2022 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

from chrisapp.base import ChrisApp
from analysis.FINAL_compute_std_by_val_resampling import Analysis_one
from analysis.FINAL_run_l1_regr_quant_function_stability_and_supp_analyses import Analysis_two
from analysis.FINAL_run_RF_homology_detection import Analysis_three
from analysis.FINAL_run_transfer_analysis_function_prediction_stability import Analysis_four

Gstr_title = r"""
             _                                  _           _     
            (_)                                | |         (_)    
 _   _ _ __  _ _ __ ___ _ __   __ _ _ __   __ _| |_   _ ___ _ ___ 
| | | | '_ \| | '__/ _ \ '_ \ / _` | '_ \ / _` | | | | / __| / __|
| |_| | | | | | | |  __/ |_) | (_| | | | | (_| | | |_| \__ \ \__ \
 \__,_|_| |_|_|_|  \___| .__/ \__,_|_| |_|\__,_|_|\__, |___/_|___/
                       | |______                   __/ |          
                       |_|______|                 |___/           
"""

Gstr_synopsis = """

(Edit this in-line help for app specifics. At a minimum, the 
flags below are supported -- in the case of DS apps, both
positional arguments <inputDir> and <outputDir>; for FS and TS apps
only <outputDir> -- and similarly for <in> <out> directories
where necessary.)

    NAME

       unirep_analysis

    SYNOPSIS

        docker run --rm fnndsc/pl-unirep_analysis unirep_analysis                     \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir> 

    BRIEF EXAMPLE

        * Bare bones execution

            docker run --rm -u $(id -u)                             \
                -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing      \
                fnndsc/pl-unirep_analysis unirep_analysis                        \
                /incoming /outgoing

    DESCRIPTION

        `unirep_analysis` ...

    ARGS

        [-h] [--help]
        If specified, show help message and exit.
        
        [--json]
        If specified, show json representation of app and exit.
        
        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.
        
        [--savejson <DIR>] 
        If specified, save json representation file to DIR and exit. 
        
        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.
        
        [--version]
        If specified, print version number and exit. 
"""


class Unirep_analysis(ChrisApp):
    """
    An app to ...
    """
    PACKAGE                 = __package__
    TITLE                   = 'A ChRIS plugin app'
    CATEGORY                = ''
    TYPE                    = 'ds'
    ICON                    = ''   # url of an icon image
    MIN_NUMBER_OF_WORKERS   = 1    # Override with the minimum number of workers as int
    MAX_NUMBER_OF_WORKERS   = 1    # Override with the maximum number of workers as int
    MIN_CPU_LIMIT           = 1000 # Override with millicore value as int (1000 millicores == 1 CPU core)
    MIN_MEMORY_LIMIT        = 200  # Override with memory MegaByte (MB) limit as int
    MIN_GPU_LIMIT           = 0    # Override with the minimum number of GPUs as int
    MAX_GPU_LIMIT           = 0    # Override with the maximum number of GPUs as int

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}

    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """
        print(Gstr_title)
        print('Version: %s' % self.get_version())
        # Output the space of CLI
        d_options = vars(options)
        for k,v in d_options.items():
            print("%20s: %-40s" % (k, v))
        print("")
        
        
        
        print("\n\n1)Running FINAL_compute_std_by_val_resampling analysis\n\n")
        try:
          Analysis_one(options.inputdir,options.outputdir)
        except Exception as err:
          print("\n\nERROR:",err)
          
          
          
        print("\n\n2)Running FINAL_run_l1_regr_quant_function_stability_and_supp_analyses analysis\n\n")
        try:
          Analysis_two(options.inputdir,options.outputdir)
        except Exception as err:
          print("\n\nERROR:",err)
           
           
        
        print("\n\n3)Running FINAL_run_RF_homology_detection analysis***\n\n")
        try:
          Analysis_three(options.inputdir,options.outputdir)
        except Exception as err:
          print("\n\nERROR:",err)
        
        
        print("\n\n4)Running FINAL_run_transfer_analysis_function_prediction_stability analysis***\n\n")
        try:
          Analysis_four(options.inputdir,options.outputdir)
        except Exception as err:
          print("\n\nERROR:",err)

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)
