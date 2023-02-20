from utils import mkdir_if_not_exists

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import torch
import random
# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error()
# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import argparse
import pdb

## CONFIG ##
ATTENTION_K = 4 # Split attention dimension into 1/K, 2/K ... K/K
TOTAL_EPOCHS = 5
USE_SHORT = False
MAX_SEQ_LEN = 128
parser = argparse.ArgumentParser(description='Progressive shrinking trainer')
parser.add_argument('--progressive_shrinking', action="store_true")
parser.add_argument('--num_subnets', nargs='+', type=int,default=[4],
                    help='number of subnets to be sampled for each epoch')
parser.add_argument('--min_encoder_num', metavar='N', type=int, default=5,
                    help='minimum encoder number for the elastic encoder layers')
parser.add_argument('--evaluate_every_epoch',action='store_true')
parser.add_argument('--epochs_per_stage',  type=int, default=1,
                    help='minimum encoder number for the elastic encoder layers')
parser.add_argument('--no_sandwich',action='store_true')
parser.add_argument('--elastic_encoder',action='store_true')
parser.add_argument('--elastic_attention', action='store_true')
parser.add_argument('--train_base_model',action='store_true' )
parser.add_argument('--train_individual_model', action='store_true')
# parser.add_argument('--attention_dimensions', nargs='+', type=float,default=None)

#######
## TODO : 
## - support glue dataset for training
## - standardize the logging and saving using the refactored codes.
## - sample mutliple models per batch
## - get averaged loss for checkpointing.
#######





class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea

def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']

def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def save_accuracy_log(save_path, train_accuracy_list, valid_accuracy_list, global_steps_list):
    if save_path == None:
        return
    
    state_dict = {'train_accuracy_list': train_accuracy_list,
                  'valid_accuracy_list': valid_accuracy_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Accuracy log saved to ==> {save_path}')
    
def load_accuracy_log(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Accuracy log loaded from <== {load_path}')
    
    return state_dict['train_accuracy_list'], state_dict['valid_accuracy_list'], state_dict['global_steps_list']

# Training Function
def train_model(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = None,
          valid_loader = None,
          num_epochs = 5,
          eval_every = None,
          file_path = None,
          file_prefix='base',
          best_valid_loss = float("Inf"),
          min_encoder_num = None,
          attention_dimensions=None,
          num_subnets = 4,
          sandwich=False,
          eval_attention=1):
    if sandwich:
        print("Sandwich training is on")
    print("a total of ", num_subnets, "subnets would be used for training")
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    
    val_acc = 0.0
    train_acc = 0.0
    valid_accuracy_list = []
    train_accuracy_list = []
    val_y_pred = []
    val_y_true = []
    train_y_pred = []
    train_y_true = []
    
    
    if min_encoder_num is None:
        min_encoder_num = model.encoder.config.num_hidden_layers
        print("because min_encoder_num is none, it is set to", min_encoder_num) 
    
    if attention_dimensions is None:
        attention_dimensions = [1]
        print("because attention_dimensions is none, it is set to", attention_dimensions) 

    print("*min_encoder_num : ", min_encoder_num)
    print("*attention dimensions : ", attention_dimensions)


    # training loop
    model.train()
    ##TODO
    change_subnet_every = len(train_iter) // num_subnets
    if change_subnet_every == 0:
        change_subnet_every = 1
    if eval_every == 0:
        eval_every = 1
   
    for epoch in range(num_epochs):
        
        ##Set the list of subnets to train
        subnet_depth_list = []
        subnet_attention_ratio_list = [] 
        subnet_counter = 0
        if sandwich:
            subnet_depth_list = [model.encoder.config.num_hidden_layers, min_encoder_num]
            ## TODO: fix this -> because number of attention heads must be able to divide the hidden layer
            ## just past in the dimension and set the range.
            subnet_attention_ratio_list = [attention_dimensions[0],attention_dimensions[-1]]
        subnet_depth_list.extend(random.choices(list(range(min_encoder_num, model.encoder.config.num_hidden_layers+1)), k=(num_subnets - len(subnet_depth_list))))
        # print(type(min_attention_heads_num))
        # print(type(model.encoder.config.num_attention_heads))
        subnet_attention_ratio_list.extend(random.choices(attention_dimensions, k=(num_subnets - len(subnet_attention_ratio_list))))
        random.shuffle(subnet_depth_list)
        random.shuffle(subnet_attention_ratio_list)
        print("subnet depths for training in epoch", epoch, " : ", subnet_depth_list)
        print("subnet attention head num for training in epoch", epoch, " : ", subnet_attention_ratio_list)

        
        
        for (text,sentiment), _ in train_loader:
            if global_step % change_subnet_every == 0 and subnet_counter < num_subnets:
                print("at global step of ", global_step, " switching the subnet depth")
                model.encoder.set_max_encoder_num(subnet_depth_list[subnet_counter])
                model.encoder.set_attention_ratio(subnet_attention_ratio_list[subnet_counter])
                print("training the model with encoder number of " , subnet_depth_list[subnet_counter])
                subnet_counter+=1
                
            
            text = text.type(torch.LongTensor)           
            text = text.to(device)
            sentiment = sentiment.type(torch.LongTensor)  
            sentiment = sentiment.to(device)
            output = model(text, sentiment)
            loss, output = output
            
            ##update the prediction and true list.
            train_y_pred.extend(torch.argmax(output, 1).tolist())
            train_y_true.extend(sentiment.tolist())
            train_acc +=accuracy_score(train_y_pred,train_y_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            
            # update running values
            running_loss += loss.item()
            global_step += 1
                    
            

            # evaluation step
            if global_step % eval_every == 0:
                ##run validation on the smallest and the largest
                
                model.eval()
                with torch.no_grad():

                    ## Change the number of encoder layers to max
                    model.encoder.set_max_encoder_num(model.encoder.config.num_hidden_layers)
                    model.encoder.set_attention_ratio(eval_attention)
                    # Evaluation done on the largest subnet only while training.
                    # validation loop
                    for (text,sentiment), _ in valid_loader:
                        text = text.type(torch.LongTensor)           
                        text = text.to(device)
                        sentiment = sentiment.type(torch.LongTensor)  
                        sentiment = sentiment.to(device)
                        output = model(text, sentiment)
                        loss, output = output
                        
                        valid_running_loss += loss.item()
                        
                        ##update the prediction and true list.
                        val_y_pred.extend(torch.argmax(output, 1).tolist())
                        val_y_true.extend(sentiment.tolist())
                        val_acc += accuracy_score(val_y_pred,val_y_true)

                    # changethe number of encoder layers back 
                    model.encoder.set_max_encoder_num(subnet_depth_list[subnet_counter-1])
                    model.encoder.set_attention_ratio(subnet_attention_ratio_list[subnet_counter-1])

                    
                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                
                average_train_accuracy = train_acc / eval_every
                average_valid_accuracy = val_acc / eval_every
                train_accuracy_list.append(average_train_accuracy)
                valid_accuracy_list.append(average_valid_accuracy)
                

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                train_acc = 0.0
                val_acc = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' +file_prefix+ '_model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + file_prefix+ '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list)


                
    save_metrics(file_path + '/' +file_prefix+ '_metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    save_accuracy_log(file_path + file_prefix+ '_acc_log.pt', train_accuracy_list, valid_accuracy_list, global_steps_list)

    print('Finished Training!')

# Evaluation Function
def evaluate(model, test_loader,encoder_num=None, attention_ratio=None ):
    y_pred = []
    y_true = []
    
    ## TODO : support num attention heads
    if encoder_num:
        model.encoder.set_max_encoder_num(encoder_num)
    else:
        encoder_num = model.encoder.config.num_hidden_layers
        model.encoder.set_max_encoder_num(model.encoder.config.num_hidden_layers)
    
    if attention_ratio:
        model.encoder.set_attention_ratio(attention_ratio)
    else:
        model.encoder.set_attention_ratio(1)

    
    print("evaluating with  encoder number of ", encoder_num)
    print("evaluating with attention layer number of ", attention_ratio)
    model.eval()
    with torch.no_grad():
        for (text,sentiment), _ in test_loader:
                text = text.type(torch.LongTensor)           
                text = text.to(device)
                sentiment = sentiment.type(torch.LongTensor)  
                sentiment = sentiment.to(device)
                output = model(text, sentiment)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(sentiment.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_pred,y_true)
    print("acc : ", acc)
    return acc

def train_progressive_shrinking(elastic_encoder=False, 
        min_encoder_num=None, 
        elastic_attention=False, 
        epochs_per_stage=1,
        attention_dimensions=None,
        num_subnets =4,
        sandwich=True,
        evaluate_every_epoch=False,
        **train_kwargs):
    if elastic_encoder:
        assert min_encoder_num, "set minimum elastic encoder number"
    if elastic_attention:
        assert attention_dimensions, "provide dimensions for the attetion head number"
        attention_dimensions.sort(reverse=True)
    
    print("\n\n progressive shrinking train started ")
    total_epochs_to_run = 0
    ## create a model to train
    model = BERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    final_accuracy=[]

    ## Do Progressive shrinking on elastic encoder number:
    if elastic_encoder:
        max_encoder_num = model.encoder.config.num_hidden_layers
        total_elastic_encoder_steps = max_encoder_num - min_encoder_num + 1
        total_epochs_to_run += total_elastic_encoder_steps * epochs_per_stage

        if evaluate_every_epoch:
            last_model_acc = np.zeros((total_elastic_encoder_steps,total_elastic_encoder_steps))
            best_model_acc = np.zeros((total_elastic_encoder_steps,total_elastic_encoder_steps))

        file_prefix= ""
        for i in range(max_encoder_num,min_encoder_num,-1):
            print("\n\n****starting with layer number of ",i,"****")
            file_prefix = "elastic_encoder_min_" + str(i) + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(num_subnets)+"_sandwich_"+str(sandwich)
            train_model(model=model, optimizer=optimizer, min_encoder_num=i,num_epochs = epochs_per_stage,num_subnets=num_subnets,sandwich=(sandwich), file_prefix=file_prefix, **train_kwargs)

            if evaluate_every_epoch:
                ## Evaluation using the last model
                for j in range(min_encoder_num,max_encoder_num+1):
                    acc = evaluate(model, test_iter, encoder_num=j)
                    last_model_acc[i-min_encoder_num][j-min_encoder_num] = acc
                    print("accuracy when using ", j, " layers : ", acc)
                print("results of the model trained using ",i," layers ")

                ## Print evaluation results
                for j in range(min_encoder_num,max_encoder_num+1):
                    print(j, " -> ", last_model_acc[i-min_encoder_num][j-min_encoder_num])

                ## Evaluation using the best model
                best_model = BERT().to(device)
                print("loading the best model")
                load_checkpoint(destination_folder +"/"+ file_prefix+ '_model.pt', best_model)
                for j in range(min_encoder_num,max_encoder_num+1):
                    acc = evaluate(best_model, test_iter, encoder_num=j)
                    best_model_acc[i-min_encoder_num][j-min_encoder_num] = acc
                    print("accuracy when using ", j, " layers : ", acc)
                print("results of the model trained using ",i," layers ")

                ## Print the results for the best model
                for j in range(min_encoder_num,max_encoder_num+1):
                    print(j, " -> ", best_model_acc[i-min_encoder_num][j-min_encoder_num])
        
        if not evaluate_every_epoch:
            last_model_acc = np.zeros(total_elastic_encoder_steps)
            best_model_acc = np.zeros(total_elastic_encoder_steps)
            ## Evaluation using the last model
            for j in range(min_encoder_num,max_encoder_num+1):
                acc = evaluate(model, test_iter, encoder_num=j)
                last_model_acc[j-min_encoder_num] = acc
                print("accuracy when using ", j, " layers : ", acc)

            ## Print evaluation results
            for j in range(min_encoder_num,max_encoder_num+1):
                print(j, " -> ", last_model_acc[j-min_encoder_num])

            ## Evaluation using the best model
            best_model = BERT().to(device)
            print("loading the best model")
            load_checkpoint(destination_folder +"/"+ file_prefix+ '_model.pt', best_model)
            for j in range(min_encoder_num,max_encoder_num+1):
                acc = evaluate(best_model, test_iter, encoder_num=j)
                best_model_acc[j-min_encoder_num] = acc
                print("accuracy when using ", j, " layers : ", acc)

            ## Print the results for the best model
            for j in range(min_encoder_num,max_encoder_num+1):
                print(j, " -> ", best_model_acc[j-min_encoder_num])
        
        file_prefix =  "elastic_encoder_min_" + min_encoder_num + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(num_subnets)+"_sandwich_"+str(sandwich)
        with open(os.path.join(output_destination_folder,"./"+file_prefix + '_last.npy'), 'wb') as f:
            np.save(f, last_model_acc)
        with open(os.path.join(output_destination_folder,"./"+file_prefix + '_best.npy'), 'wb') as f:    
            np.save(f, best_model_acc)
        if evaluate_every_epoch:
            save_heat_map("elastic encoder", list(range(min_encoder_num,min_encoder_num+1)),last_model_acc,1,True)
            save_heat_map("elastic encoder", list(range(min_encoder_num,min_encoder_num+1)),best_model_acc,1,False)

    if elastic_attention:
        print("training with elastic attention")
        total_elastic_attention_steps = len(attention_dimensions)
        total_epochs_to_run += total_elastic_attention_steps * epochs_per_stage

        if evaluate_every_epoch:
            last_model_acc = np.zeros((total_elastic_attention_steps,total_elastic_attention_steps))
            best_model_acc = np.zeros((total_elastic_attention_steps,total_elastic_attention_steps))

        file_prefix= ""
        for i, attention_ratio in enumerate(attention_dimensions):
            print("\n\n****starting with attention ratio of ",attention_ratio,"****")
            file_prefix = "elastic_attention_ratio" + str(attention_ratio) + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(num_subnets)+"_sandwich_"+str(sandwich)
            print("Training prefix : ", file_prefix)
            print("\n\n attention dimensions for this round : ",attention_dimensions[:i+1] )
            train_model(model=model, optimizer=optimizer, attention_dimensions=attention_dimensions[:i+1], 
                num_epochs = epochs_per_stage,num_subnets=num_subnets,sandwich=sandwich, 
                file_prefix=file_prefix, **train_kwargs)


            ## Evaluation using the last model
            
            for j, attention_ratio in enumerate(attention_dimensions):
                acc = evaluate(model, test_iter, attention_ratio=attention_ratio)
                last_model_acc[i][j] = acc
                print("accuracy when using ", j, " attention head lower bound : ", acc)

            print("results of the model trained using ",i," lower bounded attention heads ")

            ## Print evaluation results
            for j, attention_ratio in enumerate(attention_dimensions):
                print(attention_ratio, " -> ", last_model_acc[i][j])

            ## Evaluation using the best model
            best_model = BERT().to(device)
            print("loading the best model")
            load_checkpoint(destination_folder +"/"+ file_prefix+ '_model.pt', best_model)
            for j, attention_ratio in enumerate(attention_dimensions):
                acc = evaluate(best_model, test_iter, attention_ratio=attention_ratio)
                best_model_acc[i][j] = acc
                print("accuracy when using ", j, " attention ratio lower bound : ", acc)

            print("results of the model trained using ",i," lower bounded attention ratios ")

            ## Print the results for the best model
            for j, attention_ratio in enumerate(attention_dimensions):
                print(attention_ratio, " -> ", best_model_acc[i][j])

        file_prefix =  "elastic_attention_ratio_" + str(attention_dimensions) + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(num_subnets)+"_sandwich_"+str(sandwich)
        with open(os.path.join(output_destination_folder,"./"+file_prefix + '_last.npy'), 'wb') as f:
            np.save(f, last_model_acc)
        with open(os.path.join(output_destination_folder,"./"+file_prefix + '_best.npy'), 'wb') as f:    
            np.save(f, best_model_acc)

        save_heat_map("elastic attention ratio", attention_dimensions,last_model_acc,1,True)
        save_heat_map("elastic attention ratio", attention_dimensions,best_model_acc,1,False)

    return model, total_epochs_to_run
       
## inclusive upperbound and lowerbound
def save_heat_map(file_prefix,tick_marks, data,epochs_per_stage,last ):
    plt.clf()
    assert tick_marks, "please provide tickmarks to generate the graph"
    ax = sns.heatmap(data,  linewidth=0.5,xticklabels=tick_marks, yticklabels=tick_marks)
    # Change it to invert_yaxis for elastic encoder layers
    ax.invert_xaxis()
    ax.set_xlabel('config used for eval')
    ax.set_ylabel('lower bound for the elasticity')
    # ax.axis([5, 12,5,12])
    plot_file_name = file_prefix + "num_subnets_"+str(args.num_subnets)+ "_epochs_" + str(epochs_per_stage) +"_sandwich_"+str(not args.no_sandwich)
    plot_file_name += "last" if last else "best"
    plot_file_name += ".png"

    print("plot file name : " , plot_file_name)
    plot_title_name = file_prefix + " " + str(args.num_subnets) + " subnets ; "
    plot_title_name += "sandwich " if not args.no_sandwich else "no sandwich "
    plot_title_name += str(epochs_per_stage) + " epochs per stage "
    plot_title_name += "last" if last else "best"
    plt.title(plot_title_name)
    
    print("plot title name : " , plot_title_name)

    
    plt.savefig(os.path.join(output_destination_folder,plot_file_name))

    # ax = sns.heatmap(best_model_acc,  linewidth=0.5,xticklabels=list(range(5,13)), yticklabels=list(range(5,13)))
    # ax.invert_yaxis()
    # ax.set_xlabel('num layers used for eval')
    # ax.set_ylabel('num layers for the trained model')
    # # ax.axis([5, 12,5,12])
    # plt.title("2 subnets; no sandwich; 1 epoch per layer; best model")

    # plt.savefig(os.path.join(output_destination_folder,'./2subnet_noSandwich_1epoch_best_heatmap.png'))
    # save_checkpoint(destination_folder + '/2subnet_noSandwich_1epoch_last_model.pt', model, -1) 


# setup device, directories etc.
def setup_(use_short=False):
    # Setup directory references
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", device)
    dirname = os.getcwd()
    datadir = '../data/imdb'
    if use_short:
        datadir = '../data/imdb/short'
    source_folder = os.path.join(dirname, datadir)
    destination_folder =os.path.join(dirname,'./saved_models/')
    mkdir_if_not_exists(destination_folder)
    output_destination_folder=os.path.join(dirname,'./outputs_2')
    mkdir_if_not_exists(output_destination_folder)  
    return device, dirname, source_folder, destination_folder, output_destination_folder

def setup_data(tokenizer, source_folder, max_seq_len=128):
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    # Fields
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                       fix_length=max_seq_len, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('text', text_field),('sentiment', label_field)]
    # TabularDataset
    train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                               test='test.csv', format='CSV', fields=fields, skip_header=True)
    print("LOADED DATA:\n", "(# train, # valid, # test) = ", len(train), len(valid), len(test))    
    # Iterators
    train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                                device=device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=True, sort=False)
    return train_iter, valid_iter, test_iter

def run_progressive_shrinking(args, train_kwargs):
    print("starting progressive shrinking")
    print("num subnets : ",args.num_subnets)
    for num_subnets in args.num_subnets:
        model, total_epochs_to_run =train_progressive_shrinking(elastic_encoder=args.elastic_encoder,
            elastic_attention=args.elastic_attention,
            min_encoder_num=args.min_encoder_num,
            epochs_per_stage=args.epochs_per_stage, 
            attention_dimensions=attention_dimensions,
            evaluate_every_epoch=args.evaluate_every_epoch,
            num_subnets=num_subnets,
            sandwich=not args.no_sandwich,
            **train_kwargs)
    print("done training progressive shrinking")

def run_individual_model(args, train_kwargs, attention_dimensions, base_cases_eval=dict()):
    if args.elastic_encoder:
        pass
    # Fill this out
    if args.elastic_attention:
        assert attention_dimensions, "attention dimensions should be provided"
        evaluation_result = dict()
        for i, attention_ratio in enumerate(attention_dimensions):
            print("\n\n****starting with attention head ratio of ",attention_ratio,"****")
            file_prefix = "elastic_attention_ratio" + str(attention_ratio) + "_epochs_" + str(5) + "individual"
            model = BERT().to(device)
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
            train_model(model=model, optimizer=optimizer,attention_dimensions=[attention_ratio], 
                num_epochs =TOTAL_EPOCHS, file_prefix=file_prefix,eval_attention=attention_ratio, 
                **train_kwargs)
            best_model = BERT().to(device)
            print("loading the best model")
            load_checkpoint(destination_folder +"/"+ file_prefix+ '_model.pt', best_model)
            evaluation_result[attention_ratio] = evaluate(best_model, test_iter, attention_ratio=attention_ratio)

        print("evaluation result of individually training different attention models")
        print(evaluation_result)
        base_cases_eval['attention_individual'] = evaluation_result
    return base_cases_eval

def run_base_model(args, train_kwargs, attention_dimensions, base_cases_eval=dict()):
    base_model = BERT().to(device)
    optimizer = optim.Adam(base_model.parameters(), lr=2e-5)
    file_prefix="base"
    train_model(model=base_model, optimizer=optimizer,num_epochs =TOTAL_EPOCHS, file_prefix=file_prefix, **train_kwargs)
    if args.elastic_attention:
        base_model_evaluation_result = {}
        print("evaluating base model")
        for j, attention_ratio in enumerate(attention_dimensions):
            acc = evaluate(base_model, test_iter, attention_ratio=attention_ratio)
            base_model_evaluation_result[attention_ratio] = acc
            print("accuracy when using ", j, " attention head lower bound : ", acc)
        print(base_model_evaluation_result)
        base_cases_eval['attention_base_last'] = base_model_evaluation_result
        best_model = BERT().to(device)
        load_checkpoint(destination_folder +"/"+ file_prefix+ '_model.pt', best_model)
    
        base_model_evaluation_result = {}
        print("evaluating base model")
        for j, attention_ratio in enumerate(attention_dimensions):
            acc = evaluate(best_model, test_iter, attention_ratio=attention_ratio)
            base_model_evaluation_result[attention_ratio] = acc
            print("accuracy when using ", j, " attention head lower bound : ", acc)
        print(base_model_evaluation_result)
        base_cases_eval['attention_base_best'] = base_model_evaluation_result
    return base_cases_eval

if __name__ == '__main__':
    attention_dimensions = [i/ATTENTION_K for i in range(1,ATTENTION_K)]
    args = parser.parse_args()
    print("ARGS:", args)
    device, dirname, source_folder, destination_folder, output_destination_folder = setup_(use_short=USE_SHORT)
    # Setup tokenizer and data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_iter, valid_iter, test_iter = setup_data(tokenizer, source_folder, max_seq_len=MAX_SEQ_LEN)
    train_kwargs = {
        'train_loader': train_iter,
        'valid_loader': valid_iter,
        'file_path': destination_folder,
        'eval_every': len(train_iter) // 2,
    }
    if args.progressive_shrinking:        
        run_progressive_shrinking(args, train_kwargs)
    base_cases_eval={}
    if args.train_individual_model:
        base_cases_eval = run_individual_model(args, train_kwargs, attention_dimensions, base_cases_eval=base_cases_eval)
    if args.train_base_model:
        base_cases_eval = run_base_model(args, train_kwargs, attention_dimensions, base_cases_eval=base_cases_eval)
    print(base_cases_eval)
