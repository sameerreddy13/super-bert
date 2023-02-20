import torch
import numpy as np
from constants import *
from model import load_model
from transformers import BertConfig
from os.path import join
from data import load_data

def measure_latency(model, text, device, write_file, steps=None):
    running_time = []
    # timers
    st, et = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # measure latency for steps iterations
    with torch.no_grad():
        for step in range(steps):
            # gpu transfer and inference
            st.record()
            text = text.to(device)
            _ = model(text)
            et.record()
            # sync GPU
            torch.cuda.synchronize()
            # measure elapsed time
            inft = st.elapsed_time(et)
            # exclude warm up steps
            if step <= steps*RAMP_UP:
                continue
            # write results
            print_string="{:f}\n".format(inft)
            write_file.write(print_string)

            if VERBOSE:
                running_time.append(inft)
                # if inft > 200:
                #     print(f"Outlier step {step} with runtime of {inft}(ms)")
    if VERBOSE:
        print("AVERAGE INF TIME (ms):", np.mean(running_time))
        print("(MIN, MAX) INF TIME (ms):", (np.min(running_time), np.max(running_time)))


def measure_and_save(model, test_item, device, wp, **latency_kwargs):
    '''
    helper to measure latency and save results
    '''
    print(wp)
    write_file = open(wp, "w")
    measure_latency(model, test_item, device, write_file, **latency_kwargs)
    write_file.close()

## measure latency along a given list of models
def measure_latency_along_list(test_item, source_folder, device, log_dir, model_list, **latency_kwargs):
    for pretrained in model_list:
        model = load_model(pretrained, device)
        wp = join(".", log_dir, pretrained.split("/")[-1] + "_time_stamp.log")
        measure_and_save(model, test_item, device, wp, **latency_kwargs)
        del model

# measure latency across attention ratio
def measure_end_to_end_varying_attention_ratio(test_item, source_folder, device, log_dir, approach,attention_config,base_model, **latency_kwargs):
    pretrained = base_model
    model = load_model(pretrained, device)
    print("using elastic attention approach : ", approach)
    for attention_ratio in attention_config:
        ## set attention ratio
        print("setting attention ratio to ", attention_ratio)
        model.encoder.bert.set_attention_ratio(attention_ratio)
        wp = join(".", log_dir,pretrained+ "_attention_" + str(approach) + "_" + str(attention_ratio))+"_time_stamp.log"
        measure_and_save(model, test_item, device, wp, **latency_kwargs)

# Measure latency along BERT miniatures
def measure_latency_for_miniatures(test_item, source_folder, device, log_dir, **latency_kwargs):
    model_dict= VARY_MINIATURES
    for alias, pretrained in model_dict.items():
        model = load_model(pretrained, device)
        wp = join(".", log_dir, alias + "_time_stamp.log")
        measure_and_save(model, test_item, device, wp, **latency_kwargs)


def measure_end_to_end_varying_elastic_layer(test_item, source_folder, device, log_dir,layer_config,base_model, **latency_kwargs):
    pretrained = base_model 
    model = load_model(pretrained, device)
    for layer_num in layer_config:  
        ## set attention ratio
        print("setting max layer number to", layer_num)
        model.encoder.bert.set_max_layers(layer_num)
        wp = join(".", log_dir, pretrained+"_layer_" + str(layer_num))+"_time_stamp.log"
        measure_and_save(model, test_item, device, wp, **latency_kwargs)

def measure_end_to_end_varying_elastic_hidden(test_item, source_folder, device, log_dir, hidden_config, base_model, **latency_kwargs):
    pretrained = base_model
    model = load_model(pretrained, device)
    attention_ratio = [i/12 for i in [2,4,8,12,16]]
    for hidden_num, attention in zip(hidden_config,attention_ratio):
        print("setting hidden dimension to", hidden_num)
        model.encoder.set_hidden_dimension(hidden_num)
        ## need to change the attention ratio according to the hidden dimension
        # 2 4 8 12 16
        model.encoder.bert.set_attention_ratio(attention)
        # model.encoder.bert.set_hidden_dimension(hidden_num)
        wp = join(".", log_dir, pretrained+"_hidden_" + str(hidden_num))+"_time_stamp.log"
        measure_and_save(model, test_item, device, wp, **latency_kwargs)

# def measure_end_to_end_varying_elastic_layer(test_item, source_folder, device, log_dir,layer_config,base_model, **latency_kwargs):
#     pretrained = base_model 
#     model = load_model(pretrained, device)
#     for layer_num in layer_config:  
#         ## set attention ratio
#         print("setting hidden dimension to", layer_num)
#         model.encoder.bert.set_hidden_dim(layer_num)
#         wp = join(".", log_dir, pretrained+"_layer_" + str(layer_num))+"_time_stamp.log"
#         measure_and_save(model, test_item, device, wp, **latency_kwargs)
        
def measure_end_to_end_joint(test_item, source_folder,device,log_dir,layer_config,attention_config,hidden_config,base_model, **latency_kwargs):
    pretrained=base_model
    model = load_model(pretrained, device)

    assert len(layer_config) == len(attention_config), "the number of layer and attention configuration should match"
    custom_log_file_names = []
    for i in range(len(layer_config)):
        print("setting max layer number to", layer_sconfig[i])
        model.encoder.bert.set_max_layers(layer_config[i])
        print("setting attention ratio to ", attention_config[i])
        model.encoder.bert.set_attention_ratio(attention_config[i])
        print("setting hidden dimension to ", hidden_config[i])
        model.encoder.bert.set_hidden_dimension(hidden_config[i])
        wp = join(".", log_dir, pretrained+"_layer_" + str(layer_config[i])+"_attention_"+str(attention_config[i])+"_hidden_"+str(hidden_config[i]))+"_time_stamp.log"
        measure_and_save(model, test_item, device, wp, **latency_kwargs)


# Measure latency with varying ffn intermediate size
# TODO @sameer
def measure_latency_for_ffn_size(source_folder, device, N: int = 6):
    '''
    The default size of 3072 is split N times into (3072 * 1/N, ... 3072)
    '''
    N = 6
    sizes = [3072 * (i/N) for i in range(1, N)]
    sizes = list(map(int, sizes))
    test_iter = create_dataset(source_folder, device)
    test_item = None
    for (x, _) in test_iter:
        test_item=x

    for s in sizes:
        model_config = BertConfig(intermediate_size=s)

def measure_simulated_miniature(test_item, source_folder, device, log_dir,  base_model, **latency_kwargs):
    pretrained = base_model
    model = load_model(pretrained, device)
    attention_ratio = [i/16 for i in [2,4,8,8,12,16]]
    layer_config = [2,4,4,8,12,24]
    hidden_config=[128,256,512,512,768,1024]
    names=['tiny','mini','small','medium','base','large']
    intermediate_ratio = [0.125,0.25,0.5,0.5,.75,1.0]
    
    for hidden_num, attention,layer,name,ir in zip(hidden_config,attention_ratio,layer_config,names,intermediate_ratio):
        print("setting hidden dimension to", hidden_num)
        model.encoder.set_hidden_dimension(hidden_num)
        ## need to change the attention ratio according to the hidden dimension
        # 2 4 8 12 16
        model.encoder.bert.set_attention_ratio(attention)
        model.encoder.bert.set_max_layers(layer)
        model.encoder.bert.set_intermediate_ratio(ir)
        # model.encoder.bert.set_hidden_dimension(hidden_num)
        wp = join(".", log_dir, pretrained+"_sim_mini_" + str(name))+"_time_stamp.log"
        measure_and_save(model, test_item, device, wp, **latency_kwargs)