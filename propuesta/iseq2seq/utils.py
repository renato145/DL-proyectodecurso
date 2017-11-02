import numpy as np
import os, spacy, pickle, math, torch
from time import time
from torch.nn import functional
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets

def load_anki_dataset(path):
    spacy_es = spacy.load('es')
    spacy_en = spacy.load('en')

    def tokenize_es(text):
        return [tok.text for tok in spacy_es.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    ES = data.Field(tokenize=tokenize_es, lower=True)
    EN = data.Field(tokenize=tokenize_en, lower=True,
                    init_token='<sos>', eos_token='<eos>')

    with open(path, 'rb') as f:
        loaded = pickle.load(f)

    examples_train = loaded['train']
    examples_test = loaded['test']
    examples_val = loaded['val']

    ds_train = data.Dataset(examples_train, [('en', EN), ('es', ES)])
    ds_test = data.Dataset(examples_test, [('en', EN), ('es', ES)])
    ds_val = data.Dataset(examples_val, [('en', EN), ('es', ES)])

    ES.build_vocab(ds_train.es, min_freq=3)
    EN.build_vocab(ds_train.en, min_freq=3)

    return ds_train, ds_test, ds_val, ES, EN

def write_training_log(file, epoch, tf_ratio, train_loss, test_loss):
    lines = ''
    if not os.path.exists(file):
        lines += 'epoch,teacher_forcing_ratio,train_loss,test_loss\n'
    lines += f'{epoch},{tf_ratio},{train_loss},{test_loss}\n'
    with open(file, 'a') as f:
        f.writelines(lines)