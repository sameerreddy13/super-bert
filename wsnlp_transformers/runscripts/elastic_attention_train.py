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

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import argparse

parser = argparse.ArgumentParser(description='Progressive shrinking trainer')
parser.add_argument('--num_subnets', metavar='N', type=int, default=4,
                    help='number of subnets to be sampled for each epoch')
praser.add_argument('--no_sandwich', type=bool,action='store_true')
praser.add_argument('--elastic_encoder', type=bool,action='store_true')
# TODO : elastic_encoder=False, min_encoder_num=None, elastic_attention=False, attention_dimensions=None, evaluate_every_epoch=False, train_base_model=False, train_individual_model=False


args = parser.parse_args()


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
dirname = os.getcwd()

source_folder = os.path.join(dirname,'../data/imdb')
destination_folder =os.path.join(dirname,'./saved_models/elastic_encoder')
output_destination_folder=os.path.join(dirname,'./outputs')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
# fields = [('label', label_field), ('title', text_field), ('text', text_field), ('titletext', text_field)]
fields = [('text', text_field),('sentiment', label_field)]

# TabularDataset

train, valid, test = TabularDataset.splits(path=source_folder, train='train.csv', validation='valid.csv',
                                           test='test.csv', format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.text),
                            device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=16, device=device, train=False, shuffle=True, sort=False)

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

def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 5,
          eval_every = len(train_iter) // 2,
          file_path = destination_folder,
          file_prefix='base',
          best_valid_loss = float("Inf"),
          min_encoder_num = None,
          min_attention_heads_num=None,
          num_subnets = 4,
          sandwich=False):
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



    # training loop
    model.train()
    
    ##TODO
    change_subnet_every = len(train_iter) // num_subnets
   
    for epoch in range(num_epochs):
        
        ##Set the list of subnets to train
        subnet_depth_list = []
        subnet_counter = 0
        if sandwich:
            subnet_depth_list = [model.encoder.config.num_hidden_layers, min_encoder_num]
        subnet_depth_list.extend(random.choices(list(range(min_encoder_num, model.encoder.config.num_hidden_layers+1)), k=(num_subnets - len(subnet_depth_list))))
        random.shuffle(subnet_depth_list)
        print("subnet depths for training in epoch", epoch, " : ", subnet_depth_list)
        
        
        for (text,sentiment), _ in train_loader:
            if global_step % change_subnet_every == 0 and subnet_counter < num_subnets:
                print("at global step of ", global_step, " switching the subnet depth")
                model.encoder.set_max_encoder_num(subnet_depth_list[subnet_counter])
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

def evaluate(model, test_loader,encoder_num=None, num_attention_heads ):
    y_pred = []
    y_true = []
    
    ## TODO : support num attention heads
    if encoder_num:
        model.encoder.set_max_encoder_num(encoder_num)
    else:
        encoder_num = model.encoder.config.num_hidden_layers
        model.encoder.set_max_encoder_num(model.encoder.config.num_hidden_layers)
    
    print("evaluating with  encoder number of ", encoder_num)
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


model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)


def train_progressive_shrinking(elastic_encoder=False, 
        min_encoder_num=None, 
        elastic_attention=False, 
        epochs_per_stage=1,
        attention_dimensions=None, 
        evaluate_every_epoch=False, 
        train_base_model=False, 
        train_individual_model=False):
    if elastic_encoder:
        assert min_encoder_num, "set minimum elastic encoder number"
    if elastic_attention:
        assert attention_dimensions, "provide dimensions for the attetion head number"
    
    ## create a model to train
    model = BERT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    final_accuracy=[]

    ## Do Progressive shrinking on elastic encoder number:
    if elastic_encoder:
        max_encoder_num = model.encoder.config.num_hidden_layers
        total_elastic_encoder_steps = max_encoder_num - min_encoder_num + 1


        last_model_acc = np.zeros((total_elastic_encoder_steps,total_elastic_encoder_steps))
        best_model_acc = np.zeros((total_elastic_encoder_steps,total_elastic_encoder_steps))

        file_prefix= ""
        for i in range(max_encoder_num,min_encoder_num,-1):
            print("\n\n****starting with layer number of ",i,"****")
            file_prefix = "elastic_encoder_min_" + str(i) + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(args.num_subnets)+"_sandwich_"+str(not args.no_sandwich)
            train(model=model, optimizer=optimizer, min_encoder_num=i,num_epochs = epochs_per_stage,num_subnets=args.num_subnets,sandwich=(not args.no_sandwich), file_prefix=file_prefix)


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
            load_checkpoint(destination_folder +"/"+ file_prefix+ 'model.pt', best_model)
            for j in range(min_encoder_num,max_encoder_num+1):
                acc = evaluate(best_model, test_iter, encoder_num=j)
                best_model_acc[i-min_encoder_num][j-min_encoder_num] = acc
                print("accuracy when using ", j, " layers : ", acc)
            print("results of the model trained using ",i," layers ")

            ## Print the results for the best model
            for j in range(min_encoder_num,max_encoder_num+1):
                print(j, " -> ", best_model_acc[i-min_encoder_num][j-min_encoder_num])
        
        file_prefix =  "elastic_encoder_min_" + min_encoder_num + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(args.num_subnets)+"_sandwich_"+str(not args.no_sandwich)
        with open(os.path.join(output_destination_folder,"./"+file_prefix + '_last.npy'), 'wb') as f:
            np.save(f, last_model_acc)
        with open(os.path.join(output_destination_folder,,"./"+file_prefix + '_best.npy'), 'wb') as f:    
            np.save(f, best_model_acc)

        save_heat_map("elastic encoder", list(range(min_encoder_num,min_encoder_num+1)),last_model_acc,True)
        save_heat_map("elastic encoder", list(range(min_encoder_num,min_encoder_num+1)),best_model_acc,False)

    if elastic_attention:
        total_elastic_attention_steps = len(attention_dimensions)


        last_model_acc = np.zeros((total_elastic_attention_steps,total_elastic_attention_steps))
        best_model_acc = np.zeros((total_elastic_attention_steps,total_elastic_attention_steps))

        file_prefix= ""
        for i, num_attention_heads in enumerate(attention_dimensions):
            print("\n\n****starting with layer number of ",i,"****")
            file_prefix = "elastic_attention_head" + str(num_attention_heads) + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(args.num_subnets)+"_sandwich_"+str(not args.no_sandwich)
            train(model=model, optimizer=optimizer,min_attention_heads_num=num_attention_heads, num_epochs = epochs_per_stage,num_subnets=args.num_subnets,sandwich=(not args.no_sandwich), file_prefix=file_prefix)


            ## Evaluation using the last model
            for j, num_attention_heads in enumerate(attention_dimensions):
                acc = evaluate(model, test_iter, num_attention_heads=j)
                last_model_acc[i][j] = acc
                print("accuracy when using ", j, " attention head lower bound : ", acc)

            print("results of the model trained using ",i," lower bounded attention heads ")

            ## Print evaluation results
            for j, num_attention_heads in enumerate(attention_dimensions):
                print(j, " -> ", last_model_acc[i][j])

            ## Evaluation using the best model
            best_model = BERT().to(device)
            print("loading the best model")
            load_checkpoint(destination_folder +"/"+ file_prefix+ 'model.pt', best_model)
            for j, num_attention_heads in enumerate(attention_dimensions):
                acc = evaluate(best_model, test_iter, num_attention_heads=j)
                best_model_acc[i][j] = acc
                print("accuracy when using ", j, " attention head lower bound : ", acc)

            print("results of the model trained using ",i," lower bounded attention heads ")

            ## Print the results for the best model
            for j, num_attention_heads in enumerate(attention_dimensions):
                print(j, " -> ", best_model_acc[i][j])

        file_prefix =  "elastic_attention_head_" + str(attention_dimensions) + "_epochs_" + str(epochs_per_stage) + "_num_subnets_"+str(args.num_subnets)+"_sandwich_"+str(not args.no_sandwich)
        with open(os.path.join(output_destination_folder,"./"+file_prefix + '_last.npy'), 'wb') as f:
            np.save(f, last_model_acc)
        with open(os.path.join(output_destination_folder,,"./"+file_prefix + '_best.npy'), 'wb') as f:    
            np.save(f, best_model_acc)

        save_heat_map("elastic attention heads", attention_dimensions,last_model_acc,True)
        save_heat_map("elastic attention heads", attention_dimensions,best_model_acc,False)

        
## inclusive upperbound and lowerbound
def save_heat_map(file_prefix,tick_marks, data,epochs_per_stage,last_model ):

    assert tick_marks, "please provide tickmarks to generate the graph"
    ax = sns.heatmap(data,  linewidth=0.5,xticklabels=tick_marks, yticklabels=tick_marks)
    ax.invert_yaxis()
    ax.set_xlabel('num layers used for eval')
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


    plt.savefig(os.path.join(output_destination_folder,plot_file_name)

    ax = sns.heatmap(best_model_acc,  linewidth=0.5,xticklabels=list(range(5,13)), yticklabels=list(range(5,13)))
    ax.invert_yaxis()
    ax.set_xlabel('num layers used for eval')
    ax.set_ylabel('num layers for the trained model')
    # ax.axis([5, 12,5,12])
    plt.title("2 subnets; no sandwich; 1 epoch per layer; best model")

    plt.savefig(os.path.join(output_destination_folder,'./2subnet_noSandwich_1epoch_best_heatmap.png'))
    save_checkpoint(destination_folder + '/2subnet_noSandwich_1epoch_last_model.pt', model, -1) 


