import os, spacy, pickle
import numpy as np
from time import time
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

    ds_train = data.Dataset(examples_train, [('en', EN), ('es', ES)])
    ds_test = data.Dataset(examples_test, [('en', EN), ('es', ES)])

    ES.build_vocab(ds_train.es)
    EN.build_vocab(ds_train.en)

    return ds_train, ds_test, ES, EN
