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
model_name = 'microsoft/deberta-v3-small'

sorting = "confidence"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
myUtils.tokenizer = tokenizer

ds_name = 'corona'
model_type = 'deberta'
path = f'results/mp/{model_type}/{ds_name}/{sorting}/'
aopc = AOPC(path, tokenizer)
aopc.compare_all(verbose=False, skip = ['sorts', 'alpha', 'delta'])
