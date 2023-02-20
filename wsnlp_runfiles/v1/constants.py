from collections import OrderedDict
## CONFIG ##
DEFAULT_ID = 'default' # default subfolder for experiment results i.e outputs.../DEFAULT_ID
SEED = 42

# config for latency, some of these may be deprecated #
RAMP_UP = 0.1 # ramp up ratio. the time stamp is logged for steps in [MAX_STEPS*RAMP_UP, MAX_STEPS]
SYNC = False # disable asynchronous CUDA (useful for debugging)
VERBOSE = False # set to True to print timing info and other info for debugging
PERCENTILE = 0.95
VARY_ENCODER_LAYER=[
'google/bert_uncased_L-2_H-768_A-12','google/bert_uncased_L-4_H-768_A-12',
'google/bert_uncased_L-6_H-768_A-12','google/bert_uncased_L-8_H-768_A-12',
'google/bert_uncased_L-10_H-768_A-12','google/bert_uncased_L-12_H-768_A-12'
]
VARY_HIDDEN_DIM = ['google/bert_uncased_L-12_H-128_A-2',
'google/bert_uncased_L-12_H-256_A-4','google/bert_uncased_L-12_H-512_A-8',
'google/bert_uncased_L-12_H-768_A-12'
]
VARY_MINIATURES = OrderedDict({
	"tiny" : "google/bert_uncased_L-2_H-128_A-2","mini":"google/bert_uncased_L-4_H-256_A-4",
	"small":"google/bert_uncased_L-4_H-512_A-8", "medium":"google/bert_uncased_L-8_H-512_A-8", 
	"base" : "bert-base-uncased",
	"large" : "bert-large-uncased"
})


ELASTIC_LAYER_CONFIG = [i for i in range(6,25)]
ELASTIC_ATTENTION_CONFIG = [i/10 for i in range(1,11)]
ELASTIC_INTERMEDIATE_CONFIG = [i/10 for i in range(5,11)]
ELASTIC_HIDDEN_CONFIG = [128 * i for i in range(4, 9)]
TEST_ELASTIC_CONFIGS = [0.25,0.5,0.75]