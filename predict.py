#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import numpy as np
import string
from time import time
import time
from joblib import Memory
import os
from tqdm import tqdm
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import gc
import networkx as nx
import matplotlib.pyplot as plt

gc.collect()
torch.cuda.empty_cache()
memory = Memory(location='.cache_data', verbose=0)


# In[2]:


UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'


# In[3]:
class Arguments:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    embedding_size = 300
    corpus_size = 99  #49 9999
    batch_size = 16
    accumulate = 2
    num_workers = 8
    n = 1
    max_length = 5000
    dropout = 0.5
    lr = 1e-3
    weight_decay = 1e-4
    epoch = 10
    prefetch_factor = 10
    
    
writer_args = {
    'saved_path': './model_checkpoints/style_n1',
    'runs_name': 'FakeStyleGraph',
    'resume_snapshot': '',
    'seed': 43
}

# In[4]:


def _load_Glove(args):
    stoi = {}
    itos = {}
    embedding = []
    # Add unknown and padding token into the embedding
    stoi[UNKNOWN_TOKEN] = 0
    itos[0] = UNKNOWN_TOKEN
    embedding.append(np.random.rand(args.embedding_size))
    stoi[PADDING_TOKEN] = 1
    itos[1] = PADDING_TOKEN
    embedding.append(np.random.rand(args.embedding_size))
    
    with open('embeddings/glove.6B.{}d.txt'.format(args.embedding_size), 'r', encoding='utf8') as f:
        for idx, line in enumerate(f, start=2):
            values = line.split()
            stoi[values[0]] = idx
            itos[idx] = values[0]
            embedding.append([float(v) for v in values[1:]])
            if idx > args.corpus_size:
                break
        
        embedding = np.array(embedding, dtype=np.float32)
    return stoi, itos, embedding


# In[5]:


class FakeStyleDataset(Dataset):
    def __init__(self, data, stoi, args):
        super(FakeStyleDataset).__init__()
        nodes = [[word for word in sentence.strip().split()[:args.max_length]] for _, sentence in data]
        print(nodes)
        nodes = [[stoi[word] if word in stoi else stoi[UNKNOWN_TOKEN] for word in sentence] for sentence in nodes]
        self.nodes = nodes
        
        self.stoi = stoi
        self.n = args.n
        self.labels = [label for label, _ in data]
        
    def __getitem__(self, idx):
        neigh = []
        for j in range(len(self.nodes[idx])):
            n = []
            for i in range(-self.n, self.n + 1):
                if 0 <= i + j < len(self.nodes[idx]):
                    n.append(self.nodes[idx][i+j])
                else:
                    n.append(self.stoi[PADDING_TOKEN])
            neigh.append(n)
        return (
            torch.LongTensor(self.nodes[idx]),
            torch.LongTensor(neigh),
            self.labels[idx]
        )
    
    def __len__(self):
        return len(self.nodes)
    


# In[6]:


def _load_data(scenario, fold):
    df_data = pd.read_csv('dataset/corpusSources.tsv', sep='\t', encoding='utf-8')
    valid_data = ~df_data['content'].isna()
    df_data = df_data[valid_data][['Non-credible', 'content']]
    df_data['content'] = df_data['content'].str.lower()
    df_data['content'] = df_data['content'].str.replace('[{}]'.format(string.punctuation), ' ')
    df_data['content'] = df_data['content'].str.replace('\s+', ' ', regex=True)
    print('Finished pre-process DF data')

    # k-fold with k = 5
    df_fold = pd.read_csv('fakestyle-master/NewsStyleCorpus/foldsCV.tsv', sep='\t', encoding='utf-8')
    df_fold = df_fold[valid_data]
    fold_list = df_fold[scenario + 'CV'].to_numpy() ####################################################

    fold_idx = list(range(1, 6))
    fold_idx = fold_idx[-fold:] + fold_idx[:-fold] # Rotate the fold depending on fold input
    fold_idx = {
        'train': fold_idx[0:4],
        'val': fold_idx[4:5],
        'test': fold_idx[4:5]
    }

    included_data = np.isin(fold_list, fold_idx)

    train_data = df_data.to_numpy()[np.isin(fold_list, fold_idx['train'])]
    val_data = df_data.to_numpy()[np.isin(fold_list, fold_idx['val'])]
    test_data = df_data.to_numpy()[np.isin(fold_list, fold_idx['test'])]
    
    return train_data, val_data, test_data


# In[7]:


def _load_dataset(train_data, val_data, stoi, args):
    train_dataset = FakeStyleDataset(train_data, stoi, args)
    val_dataset = FakeStyleDataset(val_data, stoi, args)
    
    return train_dataset, val_dataset


# In[8]:


def _load_dataloader(train_dataset, val_dataset, stoi, args):
    
    def pad_collate(batch):
        node_list = []
        neighbor_list = []
        label_list = []
        for sample in batch:
            node_list.append(sample[0])
            neighbor_list.append(sample[1])
            label_list.append(sample[2])
        node_list = nn.utils.rnn.pad_sequence(node_list, batch_first=True, padding_value=stoi[PADDING_TOKEN])
        
        max_len_0 = max([s.shape[0] for s in neighbor_list])
        max_len_1 = max([s.shape[1] for s in neighbor_list])
        out_dims = (len(neighbor_list), max_len_0, max_len_1)
        out_tensor = neighbor_list[0].data.new(*out_dims).fill_(stoi[PADDING_TOKEN])
        for i, tensor in enumerate(neighbor_list):
            len_0 = tensor.size(0)
            len_1 = tensor.size(1)
            out_tensor[i, :len_0, :len_1] = tensor
        neighbor_list = out_tensor
        label_list = torch.LongTensor(label_list)
        
        return node_list, neighbor_list, label_list
    
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, prefetch_factor=args.prefetch_factor, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, prefetch_factor=args.prefetch_factor, collate_fn=pad_collate)

    return train_loader, val_loader


# In[9]:


class FakeStyleGraph(nn.Module):
    
    def __init__(self, embedding, args, stoi):
        super(FakeStyleGraph, self).__init__()
        
        adjacency_matrix = torch.Tensor(embedding.shape[0], embedding.shape[0])
        nn.init.xavier_uniform_(adjacency_matrix, gain=0.5) # To ensure the initialization is not too big
        self.adjacency_matrix = nn.Parameter(adjacency_matrix) # Smaller is better to prevent under-trained
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding), freeze=False)
        self.aggregate_rate = nn.Parameter(torch.rand(embedding.shape[0])) # To ensure the initialization is balanced enough
        
        self.dropout = nn.Dropout(args.dropout)
        self.last = nn.Linear(args.embedding_size, 2)
        
    def forward(self, nodes, neighbors, stoi, args):
        nodes_embedding = self.embedding(nodes)
        neighbors_embedding = self.embedding(neighbors)
        temp_nodes = nodes.unsqueeze(2).repeat(1, 1, neighbors.shape[-1])
        h = self.adjacency_matrix[temp_nodes, neighbors]
        # Disable weight that is just for padding
        h = h.masked_fill(temp_nodes == stoi[PADDING_TOKEN], 0)
        
        M = h.reshape(-1, 1) * neighbors_embedding.reshape(-1, args.embedding_size)
        M = self.dropout(M)
        M = M.reshape(neighbors_embedding.shape)
        M, _ = torch.max(M, dim=2)
        
        # Disable representations for padding
        message_aggregate_rate = self.aggregate_rate[nodes].masked_fill(nodes == stoi[PADDING_TOKEN], 0)
        ori_aggregate_rate = (1 - message_aggregate_rate).masked_fill(nodes == stoi[PADDING_TOKEN], 0)
        representations = message_aggregate_rate.unsqueeze(-1) * M + ori_aggregate_rate.unsqueeze(-1) * nodes_embedding
        representations = torch.sum(representations, dim=1)
        label = self.last(representations)
        return label


# In[10]:


def main(config, scenario, fold):
    args = Arguments()
    
    print('Start to load GloVE Embedding')
    stoi, itos, embedding = _load_Glove(args)

    print('Start to load data')
    train_data, val_data, test_data = _load_data(scenario, fold)
    train_dataset, val_dataset = _load_dataset(train_data, val_data, stoi, args)
    print('Training Dataset:', len(train_dataset))
    print('Evaluation Dataset:', len(val_dataset))

    train_loader, val_loader = _load_dataloader(train_dataset, val_dataset, stoi, args)

    model = FakeStyleGraph(embedding, args, stoi)
    for p in model.parameters():
        if len(p.shape) > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
        else:
            nn.init.uniform_(p, 0.0, 0.1)
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.to(args.device)

    for epoch in range(args.epoch):
        model.train()
        task = 'Train'
        train_header = '{0}/Acc. {0}/Pr.  {0}/Recall   {0}/F1       {0}/Loss'.format(task)
        train_log_template = ' '.join(
                '{:>7.4f},{:>8.4f},{:8.4f},{:12.4f},{:12.4f}'.split(','))
        train_loss = 0
        train_correct = 0
        predicted_labels, target_labels = [], []

        print('Start Training')
        # Because our VRAM is not that big, we need to simulate larger batch size
        # Using gradients accumulation
        # But batch size too small will make computation too slow
        model.zero_grad()
        for idx, (nodes, neighbors, labels) in tqdm(enumerate(train_loader)):
            nodes = nodes.to(args.device)
            neighbors = neighbors.to(args.device)
            labels = labels.to(args.device)

            out = model(nodes, neighbors, stoi, args)
            loss = loss_func(out, labels).to(args.device)
            loss = loss / args.accumulate

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if (idx+1) % args.accumulate == 0:
                optimizer.step()
                model.zero_grad()
            train_loss += loss.item()
            
            predicted_labels.extend(out.argmax(dim=1).cpu().detach().numpy())
            target_labels.extend(labels.cpu().detach().numpy())
            train_correct += (out.argmax(dim=1) == labels).sum().item()

            if (idx+1) % (100*args.accumulate) == 0:
                print('Loss', loss)
        train_acc = train_correct / len(train_dataset)
        
        predicted_labels = np.array(predicted_labels)
        target_labels = np.array(target_labels)
        
        train_accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        train_precision = metrics.precision_score(target_labels, predicted_labels, average='macro')
        train_recall = metrics.recall_score(target_labels, predicted_labels, average='macro')
        train_f1 = metrics.f1_score(target_labels, predicted_labels, average='macro')
        train_confusion_matrix = metrics.confusion_matrix(target_labels, predicted_labels)
        train_avg_loss = train_loss / len(train_loader)
        
        # Printing results to the terminal
        print(train_header)
        print(train_log_template.format(
                                train_accuracy,
                                train_precision,
                                train_recall,
                                train_f1,
                                train_avg_loss
                                ))
        print('Confusion Matrix')
        print(train_confusion_matrix)
        
        task = 'Train'
        
        # Logging results into tensorboard
        config['writer'].add_scalar('Accuracy/{}'.format(task), train_accuracy, epoch+1)
        config['writer'].add_scalar('Precision/{}'.format(task), train_precision, epoch+1)
        config['writer'].add_scalar('Recall/{}'.format(task), train_recall, epoch+1)
        config['writer'].add_scalar('F1-Score/{}'.format(task), train_f1, epoch+1)
        config['writer'].add_scalar('Average Loss/{}'.format(task), train_avg_loss, epoch+1)
        gc.collect()
        torch.cuda.empty_cache()

        print('Start Evaluation')
        model.eval()
        task = 'Validation'
        validation_header = '{0}/Acc. {0}/Pr.  {0}/Recall   {0}/F1       {0}/Loss'.format(task)
        validation_log_template = ' '.join(
                '{:>7.4f},{:>8.4f},{:8.4f},{:12.4f},{:12.4f}'.split(','))
        val_loss = 0
        val_correct = 0
        predicted_labels, target_labels = [], []
        
        for idx, (nodes, neighbors, labels) in tqdm(enumerate(val_loader)):
            nodes = nodes.to(args.device)
            neighbors = neighbors.to(args.device)
            labels = labels.to(args.device)

            out = model(nodes, neighbors, stoi, args)
            loss = loss_func(out, labels).to(args.device)
            val_loss += loss.item()
            
            predicted_labels.extend(out.argmax(dim=1).cpu().detach().numpy())
            target_labels.extend(labels.cpu().detach().numpy())
            val_correct += (out.argmax(dim=1) == labels).sum().item()
        val_acc = val_correct / len(val_dataset)

        predicted_labels = np.array(predicted_labels)
        target_labels = np.array(target_labels)
        
        validation_accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        validation_precision = metrics.precision_score(target_labels, predicted_labels, average='macro')
        validation_recall = metrics.recall_score(target_labels, predicted_labels, average='macro')
        validation_f1 = metrics.f1_score(target_labels, predicted_labels, average='macro')
        validation_confusion_matrix = metrics.confusion_matrix(target_labels, predicted_labels)
        validation_avg_loss = val_loss / len(val_loader)
        
        # Printing results to the terminal
        print(validation_header)
        print(validation_log_template.format(
                                validation_accuracy,
                                validation_precision,
                                validation_recall,
                                validation_f1,
                                validation_avg_loss
                                ))
        print('Confusion Matrix')
        print(validation_confusion_matrix)
        
        print('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch+1, train_loss, val_loss, train_acc, val_acc))
       
        # Logging results into tensorboard
        config['writer'].add_scalar('Accuracy/{}'.format(task), validation_accuracy, epoch+1)
        config['writer'].add_scalar('Precision/{}'.format(task), validation_precision, epoch+1)
        config['writer'].add_scalar('Recall/{}'.format(task), validation_recall, epoch+1)
        config['writer'].add_scalar('F1-Score/{}'.format(task), validation_f1, epoch+1)
        config['writer'].add_scalar('Average Loss/{}'.format(task), validation_avg_loss, epoch+1)
        gc.collect()
        torch.cuda.empty_cache()
    model.save()
# In[11]:

#def initialize():


def test(txt,node_number):
    args = Arguments()
    args.corpus_size=int(node_number)-1
    print(args.corpus_size)
    
    print('Start to load GloVE Embedding')
    stoi, itos, embedding = _load_Glove(args)
    
    print('Start to load model')
    model = FakeStyleGraph(embedding, args, stoi)
    if node_number=='50':
        model.load_state_dict(torch.load('2022-02-07 13_28_14.805794_epoch_9.pt'))
    elif node_number=='250':
        model.load_state_dict(torch.load('2022-02-18 17_12_15.489411_epoch_9.pt'))
    elif node_number=='500':
        model.load_state_dict(torch.load('2022-02-19 21_31_52.065076_epoch_9.pt'))
    elif node_number=='1000':
        model.load_state_dict(torch.load('2022-02-20 04_00_54.761676_epoch_9.pt'))
    else:
        model.load_state_dict(torch.load('2021-09-21 08_39_35.619066_epoch_9.pt'))
    model.to(args.device)
    
    print('Start to load data')
    train_data = [[1,txt]]
    val_data = [[1,txt]]
    train_dataset, val_dataset = _load_dataset(train_data, val_data, stoi, args)
    print('Training Dataset:', len(train_dataset))
    print('Evaluation Dataset:', len(val_dataset))

    train_loader, val_loader = _load_dataloader(train_dataset, val_dataset, stoi, args)
    
    print('Start Evaluation')
    model.eval()
    
    for idx, (nodes, neighbors, labels) in enumerate(val_loader):
        nodes = nodes.to(args.device)
        neighbors = neighbors.to(args.device)
        labels = labels.to(args.device)
        
        print(nodes)
        print(neighbors)
        out = model(nodes, neighbors, stoi, args)
        visualization(txt, nodes, neighbors)
        return nodes,out.argmax(dim=1).cpu().detach().numpy()
        
   
def visualization(txt, nodes, neighbors):
    G = nx.Graph() 
    
    word_list=txt.strip().split()
    nodes=nodes.cpu().numpy().tolist()
    neighbors=neighbors.cpu().numpy().tolist()
    had_UNK=False
    for index,node in enumerate(nodes[0]):
        if node!=0:G.add_node(word_list[index])
        elif node==0 and had_UNK==False:
            G.add_node("<UNK>")
            had_UNK=True
    for index,each_node in enumerate(neighbors[0]):
        for neighbor in each_node:
            if neighbor!=1:
                if neighbor!=0:
                    neighbor_index=nodes[0].index(neighbor)
                    if nodes[0][index]!=neighbor:
                        if nodes[0][index]!=0:G.add_edge(word_list[index],word_list[neighbor_index])
                        else:G.add_edge("<UNK>",word_list[neighbor_index])
                else:
                    if nodes[0][index]!=neighbor:
                        if nodes[0][index]!=0:G.add_edge(word_list[index],"<UNK>")
                        else:G.add_edge("<UNK>","<UNK>")
    color_map = ['red' if node == "<UNK>" else '#A0CBE2' for node in G] 
    nx.draw_networkx(G,node_color=color_map)
    #plt.show()
    plt.title("Word Graph")
    plt.savefig("word_graph.png")
    plt.close()

#txt=input()
#test(txt)
