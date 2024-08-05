# -*- coding: utf-8 -*-
"""bilst.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mJf4PWU10Mr9lxWI1Oa3YTtVBvU3PjBW
"""

!pip install datasets
!pip install transformers

import datasets as ds
from datasets import load_dataset
from transformers import AutoTokenizer

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART", add_prefix_space=True)

# Dataset parameters
dataset_full_name = "midas/inspec"
dataset_subset = "raw"
dataset_document_column = "document"

keyphrase_sep_token = ";"

def preprocess_keyphrases(text_ids, kp_list):
    kp_order_list = []
    kp_set = set(kp_list)
    text = tokenizer.decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    text = text.lower()
    for kp in kp_set:
        kp = kp.strip()
        kp_index = text.find(kp.lower())
        kp_order_list.append((kp_index, kp))

    kp_order_list.sort()
    present_kp, absent_kp = [], []

    for kp_index, kp in kp_order_list:
        if kp_index < 0:
            absent_kp.append(kp)
        else:
            present_kp.append(kp)
    return present_kp, absent_kp


def preprocess_fuction(samples):
    processed_samples = {"input_ids": [], "attention_mask": [], "labels": []}
    for i, sample in enumerate(samples[dataset_document_column]):
        input_text = " ".join(sample)
        inputs = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
        )
        present_kp, absent_kp = preprocess_keyphrases(
            text_ids=inputs["input_ids"],
            kp_list=samples["extractive_keyphrases"][i]
            + samples["abstractive_keyphrases"][i],
        )
        keyphrases = present_kp
        keyphrases += absent_kp

        target_text = f" {keyphrase_sep_token} ".join(keyphrases)

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                target_text, max_length=40, padding="max_length", truncation=True
            )
            targets["input_ids"] = [
                (t if t != tokenizer.pad_token_id else -100)
                for t in targets["input_ids"]
            ]
        for key in inputs.keys():
            processed_samples[key].append(inputs[key])
        processed_samples["labels"].append(targets["input_ids"])
    return processed_samples

# Load dataset
dataset = load_dataset(dataset_full_name, dataset_subset)
# Preprocess dataset
tokenized_dataset = dataset.map(preprocess_fuction, batched=True)

import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
train=tokenized_dataset['train']
test=tokenized_dataset['test']
val=tokenized_dataset['validation']

train_document=train['document']
train_keyphrase=train['extractive_keyphrases']
train_tags=train['doc_bio_tags']


test_document=test['document']
test_keyphrase=test['extractive_keyphrases']
test_tags=test['doc_bio_tags']

val_document=val['document']
val_keyphrase=val['extractive_keyphrases']
val_tags=val['doc_bio_tags']
def yield_tokens(tokenlist):
    for tokens in tokenlist:
        yield tokens
def get_data(dataset, vocab):
    data = []
    for sentence in dataset:
            tokens = [vocab[token] for token in sentence]
            data.append(torch.LongTensor(tokens))
    return data
train_vocab=build_vocab_from_iterator(yield_tokens(train_document),min_freq=1,specials=["<UNK>"])
train_vocab.set_default_index(0)
data_train = get_data(train_document,train_vocab)
tag_vocab=build_vocab_from_iterator(yield_tokens(train_tags),min_freq=1)
data_tag=get_data(train_tags,tag_vocab)


data_val = get_data(val_document,train_vocab)
data_val_tag=get_data(val_tags,tag_vocab)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Encoder(nn.Module):
    def __init__(self,n_hidden,embedding_dim,vocab_size,tag_size,n_layers=1):
        super(Encoder, self).__init__()
        self.n_hidden=n_hidden
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        self.lstm_2 = nn.LSTM(embedding_dim+2*n_hidden, n_hidden, bidirectional=True)
        self.out = nn.Linear(2*n_hidden, tag_size)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data # context : [batch_size, n_hidden * num_directions(=2)]

    def forward(self, X):
        X = self.embedding(X) # input : [batch_size, len_seq, embedding_dim]

        input = X.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        hidden_state = Variable(torch.zeros(1*2, len(X), self.n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = Variable(torch.zeros(1*2, len(X), self.n_hidden)) # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2) # output : [batch_size, len_seq, n_hidden]
        attn_output, attention = self.attention_net(output, final_hidden_state)
        encoding=X*attention.view(1,X.shape[1],1)
        enc=torch.cat([attn_output.repeat(1,X.shape[1],1),X],dim=2)
        
        attn_context,_=self.lstm_2(enc)
        
        return  self.out(attn_context).view(-1,3)# model : [batch_size, num_classes], attention : [batch_size, n_step]

from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 2
LEARNING_RATE=0.0001
NUMBER_OF_LAYERS=2
model = Encoder(HIDDEN_DIM,EMBEDDING_DIM,train_vocab.__len__(), tag_vocab.__len__(),NUMBER_OF_LAYERS)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("EMBEDDING DIMENSION",EMBEDDING_DIM)
print("HIDDEN DIMENSION",HIDDEN_DIM)
print("EPOCHS",EPOCHS)

print("LEARNING RATE",LEARNING_RATE)
print("NUMBER OF LAYERS",NUMBER_OF_LAYERS)

for epoch in range(EPOCHS):
    l=0
    acc=0
    tags=[]
    y_true=[]
    for i in tqdm(range(len(train_document))):
        model.zero_grad()
        tag_scores = model(data_train[i].view(1,-1))
        loss = loss_function(tag_scores, data_tag[i])
        indices = torch.max(tag_scores, 1)[1]
        tags+=list(indices.detach().numpy())
        y_true+=list(data_tag[i].detach().numpy())
        loss.backward()
        optimizer.step()
    print(classification_report(y_true,tags))
with torch.no_grad():
      model.eval()
      tags=[]
      y_true=[]
      for i in tqdm(range(len(val_document))):
          tag_scores = model(data_val[i].view(1,-1))
          loss = loss_function(tag_scores, data_val_tag[i])
          indices = torch.max(tag_scores, 1)[1]
          tags+=list(indices.detach().numpy())
          y_true+=list(data_val_tag[i].detach().numpy())
          
      print(classification_report(y_true,tags))

