import warnings
import pickle
import myUtils
import os
from myUtils import *
from models.utils import *
from AOPC import *
from score import ScoreUtils
from models.dataset_loader import *
import sys
sys.path.append('models')

warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch._C._jit_set_texpr_fuser_enabled(False)
model_name = 'huawei-noah/TinyBERT_General_4L_312D'
#model_name = 'microsoft/deberta-v3-small'

sorting = "confidence"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
myUtils.tokenizer = tokenizer

#model_type = 'deberta'
model_type = 'tinybert'

legends = ['root', 'homogenity', 'probabilistic α=0.5', 'probabilistic α=0.95', 'avg', 'avg_4e-3']
aggregations = ['root_', 'homogenity_', '', '', 'avg_', f'avg_{4e-3}_']
alphas = [None, None, 0.5, 0.95, None, None]


for ds_name in ['corona', 'toy-spam', 'home-spam', 'dilemma']:
    path = f'results/mp/{model_type}/{ds_name}/{sorting}/'
    
    #aopc = AOPC(path, tokenizer)
    #aopc.compare_all(only = ['percents time', 'aopc time', 'aopc aggregation time'], agg_params = (aggregations, alphas, legends))

    aopc = AOPC(path, tokenizer, base_opt = 'stop-words')
    aopc.compare_all(only = ['percents time', 'aopc time', 'aopc aggregation time'], agg_params = (aggregations, alphas, legends))
