"""
Reproduce Supp. Fig. 13

UniRep is a generative model of protein sequences. Because this notebook requires the UniRep model, please download the unirep github into the parent directory of unirep_anlysis, including the weight files. Additionally, you must activate the unirep model environment NOT the standard analysis environment for this to run.

Runs on a 16G RAM laptop in a few minutes, no GPU

"""

import tensorflow as tf
import numpy as np
import sys
import random
import os
# To allow imports from common directory

sys.path.append('../../../unirep/')
sys.path.append('../')
from unirep import babbler1900
random.seed(42)
np.random.seed(42)
tf.set_random_seed(42)



# initialize the babbler
b = babbler1900(model_path="../../../unirep/weights/1900_weights/")

# ABC transporter, ATP binding protein (glucose) 1-223 (all here)
# http://scop2.mrc-lmb.cam.ac.uk/rep-34668.html
# PDB:1oxx
seed = """
MVRIIVKNVSKVFKKGKVVALDNVNINIENGERFGILGPSGAGKTTFMRIIAGLDVPSTG
ELYFDDRLVASNGKLIVPPEDRKIGMVFQTWALYPNLTAFENIAFPLTNMKMSKEEIRKR
VEEVAKILDIHHVLNHFPRELSGGQQQRVALARALVKDPSLLLLDEPFSNLDARMRDSAR
ALVKEVQSRLGVTLLVVSHDPADIFAIADRVGVLVKGKLVQVGKPEDLYDNPVSIQVASL
IGEINELEGKVTNEGVVIGSLRFPVSVSSDRAIIGIRPEDVKLSKDVIKDDSWILVGKGK
VKVIGYQGGLFRITITPLDSEEEIFTYSDHPIHSGEEVLVYVRKDKIKVFEKN
""".replace("\n","")
seed

babbled = b.get_babble(seed=seed[:15], length=400, temp=.5)
print(babbled)
babbled.split('stop')[0]
