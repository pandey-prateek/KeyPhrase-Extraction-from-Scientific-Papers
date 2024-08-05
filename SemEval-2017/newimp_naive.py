import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from torchtext.vocab import Vocab, vocab
import itertools
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import itertools
import os
torch.manual_seed(1)

class BiLSTMCRF(nn.Module):
    def __init__(self, sent_vocab, tag_vocab, dropout_rate, embed_size, hidden_size):
      
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab = tag_vocab
        self.embedding = nn.Embedding(len(sent_vocab), embed_size)

        self.hidden2emit_score = nn.Linear(hidden_size * 2, len(self.tag_vocab))
        self.transition = nn.Parameter(torch.randn(len(self.tag_vocab), len(self.tag_vocab)))  
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')

    def forward(self, sentences, tags, sen_lengths):

        mask = (sentences != self.sent_vocab['<PAD>']).to(self.device)  
        sentences = sentences.transpose(0, 1)  
        sentences = self.embedding(sentences)  
        emit_score = self.encode(sentences, sen_lengths)  
        loss = self.calculate_loss(tags, mask, emit_score)  
        return loss

    def encode(self, sentences, sent_lengths):

        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)  
        emit_score = self.hidden2emit_score(hidden_states)  
        emit_score = self.dropout(emit_score)  
        return emit_score

    def calculate_loss(self, tags, mask, emit_score):

        _, sent_len = tags.shape

        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  
        
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  
            log_sum = log_sum - max_v  
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  
        max_d = d.max(dim=-1)[0]  
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  
        llk = total_score - d  
        loss = -llk  
        return loss
      
    def predict(self, sentences, sen_lengths):

        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab['<PAD>'])  
        sentences = sentences.transpose(0, 1)  
        sentences = self.embedding(sentences)  
        emit_score = self.encode(sentences, sen_lengths)  
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  
        d = d.squeeze(dim=1)  
        _, max_idx = torch.max(d, dim=1)  
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab': self.tag_vocab,
            'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(params['sent_vocab'], params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

def read_corpus(filepath):

    with open (filepath, 'r') as file:
      data = file.read ()

    data = data.split ('\n')
    X = []
    y = []

    for data in data:
      data = data.split ('\t')

      if data [0] != '':
        X.append (data [0])
        
      if data [-1] != '':
        y.append (data [-1])

    X = [word.lower () for word in X]

    newX = []
    newY = []

    sentence = []
    tags = []
    
    for i in range (0, len (X)):
      if X [i] == '.':
        tags.append (y [i])
        sentence.append (X [i])
        newX.append (sentence)
        newY.append (tags)
        sentence = []
        tags = []
      else:
        sentence.append (X [i])
        tags.append (y [i])

    newX = [' '.join (x) for x in newX]
    newY = [' '.join (y) for y in newY]

    # newX = ['<START> '+x+' <END>' for x in newX]
    # newY = ['<START> '+y+' <END>' for y in newY]
    
    return newX, newY


def generate_train_dev_dataset(filepath, sent_vocab, tag_vocab, train_proportion=0.8):

    sentences, tags = read_corpus(filepath)

    sentences = words2indices(sentences, sent_vocab)

    tags = words2indices(tags, tag_vocab)

    print (tag_vocab.get_stoi ())
    data = list(zip(sentences, tags))
    random.shuffle(data)
    n_train = int(len(data) * train_proportion)
    train_data, dev_data = data[: n_train], data[n_train:]
    return train_data, dev_data


def batch_iter(data, batch_size=32, shuffle=True):

    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags = [x[1] for x in batch]
        yield sentences, tags


def words2indices(origin, vocab):

    result = [[vocab [w] for w in sent.split ()] for sent in origin]
    return result

def pad(data, padded_token, device):
  
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths

def create_vocab (tokens):
    tokens = [t.split () for t in tokens]
    tokens = list (itertools.chain (*tokens))

    tokens = list (set (tokens))
    
    v = OrderedDict ()

    for i in tokens:
      v [i] = len (v) + 2

    vocabulary = vocab (v, specials = ['<UNK>', '<PAD>'])

    vocabulary.set_default_index (vocabulary ['<UNK>'])

    return vocabulary

def init_vocabs ():
    sent, tags = read_corpus ('train.txt')

    sent_vocab = create_vocab (sent)
    tag_vocab = create_vocab (tags)

    return sent_vocab, tag_vocab

def train():
    sent_vocab, tag_vocab = init_vocabs ()
    train_data, dev_data = generate_train_dev_dataset('train.txt', sent_vocab, tag_vocab)

    # print (train_data)

    max_epoch = 15
    valid_freq = 20
    model_save_path = './model.pth'
    optimizer_save_path = './optim.pth'
    min_dev_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
    dropout_rate = 0.5
    embed_size = 300
    hidden_size = 300
    batch_size = 128
    max_clip_norm = 5.0
    lr_decay = 0.5

    model = BiLSTMCRF(sent_vocab, tag_vocab, dropout_rate, embed_size, hidden_size).to(device)

    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_iter = 0  # train iter num

    print('Training...')
    for epoch in tqdm (range(max_epoch)):
        for sentences, tags in batch_iter(train_data, batch_size=batch_size):
            train_iter += 1
            sentences, sent_lengths = pad(sentences, sent_vocab['<PAD>'], device)
            tags, _ = pad(tags, tag_vocab['<PAD>'], device)

            optimizer.zero_grad()
            batch_loss = model(sentences, tags, sent_lengths)  
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_clip_norm)
            optimizer.step()

            if train_iter % valid_freq == 0:
                dev_loss = calculate_dev_loss(model, dev_data, 64, sent_vocab, tag_vocab, device)
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss
                    model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                else:
                    lr = optimizer.param_groups[0]['lr'] * lr_decay
                    model = BiLSTMCRF.load(model_save_path, device)
                    optimizer.load_state_dict(torch.load(optimizer_save_path))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr


def test():

    sent_vocab, tag_vocab = init_vocabs ()
    sentences, tags = read_corpus('test.txt')
    sentences = words2indices(sentences, sent_vocab)
    tags = words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags))

    device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
    model = BiLSTMCRF.load('./model.pth', device)

    batch_size = 128

    predictions = []
    model.eval()

    y_true = []
    with torch.no_grad():
        for sentences, tags in batch_iter(test_data, batch_size=int(batch_size), shuffle=False):
            padded_sentences, sent_lengths = pad(sentences, sent_vocab['<PAD>'], device)
            predicted_tags = model.predict(padded_sentences, sent_lengths)

            predictions.append (predicted_tags)
            y_true.append (tags)

    return predictions, y_true


def calculate_dev_loss(model, dev_data, batch_size, sent_vocab, tag_vocab, device):

    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags in batch_iter(dev_data, batch_size, shuffle=False):
            sentences, sent_lengths = pad(sentences, sent_vocab['<PAD>'], device)
            tags, _ = pad(tags, tag_vocab['<PAD>'], device)
            batch_loss = model(sentences, tags, sent_lengths)  
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences

if os.path.exists ('model.pth'):
    pass
else:
    train ()

y_hat, y_true = test ()

y_hat = list (itertools.chain (*list (itertools.chain (*y_hat))))
y_true = list (itertools.chain (*list (itertools.chain (*y_true))))


print (accuracy_score (y_hat, y_true))

sent, tags = init_vocabs()
print (tags.get_stoi ())

print (classification_report (y_hat, y_true))