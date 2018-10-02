import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter, defaultdict
import pickle
import tensorflow as tf
import argparse
import os, sys
import time, datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
enc = OneHotEncoder()
le = LabelEncoder()


###################################################################################################################################

#Hyperparams

#MODE = "Sentiment"
MODE = "Emotion"

print("loading data...",)
x = pickle.load(open("data/pickles/data_{}.p".format(MODE.lower()),"rb"))
revs, W, word_idx_map, vocab, max_l, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
print("data loaded!") # Load data
NUM_CLASSES = len(label_index)
print(label_index)

max_l=50


###################################################################################################################################

# Loading the dataset
def get_word_indices(data_x):
    length = len(data_x.split())
    return np.asarray([word_idx_map[word] for word in data_x.split()] + [0]*(max_l-length))[:max_l]

train_data, val_data, test_data = {},{},{}
for i in range(len(revs)):
    if revs[i]['split']=="train":
        train_data[revs[i]['dialog']+"_"+revs[i]['utterance']]=(get_word_indices(revs[i]['text']),label_index[revs[i]['y']])
    elif revs[i]['split']=="val":
        val_data[revs[i]['dialog']+"_"+revs[i]['utterance']]=(get_word_indices(revs[i]['text']),label_index[revs[i]['y']])
    elif revs[i]['split']=="test":
        test_data[revs[i]['dialog']+"_"+revs[i]['utterance']]=(get_word_indices(revs[i]['text']),label_index[revs[i]['y']])

def get_dialogue_ids(keys):
    ids=defaultdict(list)
    for key in keys:
        ids[key.split("_")[0]].append(key.split("_")[1])
    return ids

# dialogue ids
train_dialogue_ids=get_dialogue_ids(train_data.keys())
val_dialogue_ids=get_dialogue_ids(val_data.keys())
test_dialogue_ids=get_dialogue_ids(test_data.keys())

def get_max_utts(train_ids, val_ids, test_ids):
    max_utts_train = max([len(train_ids[vid]) for vid in train_ids.keys()])
    max_utts_val = max([len(val_ids[vid]) for vid in val_ids.keys()])
    max_utts_test = max([len(test_ids[vid]) for vid in test_ids.keys()])
    return np.max([max_utts_train, max_utts_val, max_utts_test])

max_utts = get_max_utts(train_dialogue_ids, val_dialogue_ids, test_dialogue_ids)
###################################################################################################################################

# def make_dialogue_level(train_data, val_data, test_data, train_audio_ids, val_audio_ids, test_audio_ids):

#     train_dialogue_id, val_dialogue_id, test_dialogue_id=defaultdict(list), defaultdict(list), defaultdict(list)
#     for ID in train_data.keys():
#         if ID in train_audio_ids:
#             train_dialogue_id[ID.split("_")[0]].append(int(ID.split("_")[1]))
#     for ID in val_data.keys():
#         if ID in val_audio_ids:
#             val_dialogue_id[ID.split("_")[0]].append(int(ID.split("_")[1]))
#     for ID in test_data.keys():
#         if ID in test_audio_ids:
#             test_dialogue_id[ID.split("_")[0]].append(int(ID.split("_")[1]))

#     # Sorting the indices
#     for key, value in train_dialogue_id.items():
#         train_dialogue_id[key]=sorted(value)
#     for key, value in val_dialogue_id.items():
#         val_dialogue_id[key]=sorted(value)
#     for key, value in test_dialogue_id.items():
#         test_dialogue_id[key]=sorted(value)
#     return train_dialogue_id, val_dialogue_id, test_dialogue_id

def get_one_hot(label):
    label_arr = [0]*NUM_CLASSES
    label_arr[label]=1
    return label_arr[:]



#########################################################################################################################################################

def get_dialogue_audio_embs(train_dialogue_id, val_dialogue_id, test_dialogue_id, 
                            train_audio_emb, val_audio_emb, test_audio_emb):
    key = list(train_audio_emb.keys())[0]
    pad = [0]*len(train_audio_emb[key])

    def get_emb(dialogue_id, audio_emb):
        dialogue_audio=[]
        for vid in dialogue_id.keys():
            local_audio=[]
            for utt in dialogue_id[vid]:
                try:
                    local_audio.append(audio_emb[vid+"_"+str(utt)][:])
                except:
                    print(vid+"_"+str(utt))
                    local_audio.append(pad[:])
            for _ in range(max_utts-len(local_audio)):
                local_audio.append(pad[:])
            dialogue_audio.append(local_audio[:max_utts])
        return dialogue_audio

    train_dialogue_audio = get_emb(train_dialogue_id, train_audio_emb)
    val_dialogue_audio = get_emb(val_dialogue_id, val_audio_emb)
    test_dialogue_audio = get_emb(test_dialogue_id, test_audio_emb)

    return np.asarray(train_dialogue_audio), np.asarray(val_dialogue_audio), np.asarray(test_dialogue_audio)

def get_dialogue_lengths(train_ids, val_ids, test_ids):
    train_dialogue_length, val_dialogue_length, test_dialogue_length=[], [], []
    for vid, utts in train_ids.items():
        train_dialogue_length.append(len(utts))
    for vid, utts in val_ids.items():
        val_dialogue_length.append(len(utts))
    for vid, utts in test_ids.items():
        test_dialogue_length.append(len(utts))
    return train_dialogue_length, val_dialogue_length, test_dialogue_length

def get_dialogue_labels(train_ids, val_ids, test_ids):

    def get_labels(ids, data):
        dialogue_label=[]
        for vid, utts in ids.items():
            local_labels=[]
            for utt in utts:
                local_labels.append(get_one_hot(data[vid+"_"+str(utt)][1]))
            for _ in range(max_utts-len(local_labels)):
                local_labels.append(get_one_hot(1)) # Dummy label
            dialogue_label.append(local_labels[:max_utts])
        return dialogue_label
    train_dialogue_label=get_labels(train_ids, train_data)
    val_dialogue_label=get_labels(val_ids, val_data)
    test_dialogue_label=get_labels(test_ids, test_data)

    return np.asarray(train_dialogue_label), np.asarray(val_dialogue_label), np.asarray(test_dialogue_label)

def get_masks(train_length, val_length, test_length):

    # Creating mask
    train_mask = np.zeros((len(train_length), max_utts), dtype='float')
    for i in range(len(train_length)):
        train_mask[i,:train_length[i]]=1.0
    val_mask = np.zeros((len(val_length), max_utts), dtype='float')
    for i in range(len(val_length)):
        val_mask[i,:val_length[i]]=1.0
    test_mask = np.zeros((len(test_length), max_utts), dtype='float')
    for i in range(len(test_length)):
        test_mask[i,:test_length[i]]=1.0
    return train_mask, val_mask, test_mask

def load_audio_data():

    #Load audio Data
    AUDIO_PATH = "../data/pickles/audio_embeddings_feature_selection_{}.pkl".format(MODE.lower())
    train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(AUDIO_PATH,"rb"), encoding='latin1')
    
    train_dialogue_audio, val_dialogue_audio, test_dialogue_audio=get_dialogue_audio_embs(train_dialogue_ids, \
                                val_dialogue_ids, test_dialogue_ids, train_audio_emb, val_audio_emb, test_audio_emb)

    train_dialogue_length, val_dialogue_length, test_dialogue_length=get_dialogue_lengths(train_dialogue_ids, val_dialogue_ids, test_dialogue_ids)
    train_dialogue_labels, val_dialogue_labels, test_dialogue_labels=get_dialogue_labels(train_dialogue_ids, val_dialogue_ids, test_dialogue_ids)
    train_mask, val_mask, test_mask = get_masks(train_dialogue_length, val_dialogue_length, test_dialogue_length)

    assert(train_dialogue_labels.shape[0] == train_dialogue_audio.shape[0])
    assert(val_dialogue_labels.shape[0] == val_dialogue_audio.shape[0])
    assert(test_dialogue_labels.shape[0] == test_dialogue_audio.shape[0])

    return train_dialogue_audio, val_dialogue_audio, test_dialogue_audio, train_dialogue_labels, val_dialogue_labels, test_dialogue_labels, \
           train_mask, val_mask, test_mask, train_dialogue_ids.keys(), val_dialogue_ids.keys(), test_dialogue_ids.keys(), MODE


#########################################################################################################################################################


def get_dialogue_text_ids(train_dialogue_id, val_dialogue_id, test_dialogue_id, train_data, val_data, test_data):

    max_utts = get_max_utts(train_dialogue_id, val_dialogue_id, test_dialogue_id)
    key = list(train_data.keys())[0]
    pad = [0]*len(train_data[key][0])
    
    train_dialogue_text, val_dialogue_text, test_dialogue_text=[], [], []
    train_dialogue_length, val_dialogue_length, test_dialogue_length=[], [], []
    train_dialogue_label, val_dialogue_label, test_dialogue_label=[], [], []
    train_dialogue_name, val_dialogue_name, test_dialogue_name=[], [], []
    
    def get_values(local_data, dialogue_id, ):
        dialogue_name, dialogue_length, dialogue_text, dialogue_label=[], [], [], []
        for vid in dialogue_id.keys():
            dialogue_name.append(vid)
            local_text, local_labels=[],[]
            for utt in dialogue_id[vid]:
                local_text.append(local_data[vid+"_"+str(utt)][0][:])
                local_labels.append(get_one_hot(local_data[vid+"_"+str(utt)][1]))
            dialogue_length.append(len(local_text))
            for _ in range(max_utts-len(local_text)):
                local_text.append(pad[:])
                local_labels.append(get_one_hot(1)) # Dummy label
            dialogue_text.append(local_text[:max_utts])
            dialogue_label.append(local_labels[:max_utts])
        return dialogue_name, dialogue_length, np.asarray(dialogue_text), np.asarray(dialogue_label)


    train_dialogue_name, train_dialogue_length, train_dialogue_text, train_dialogue_label= get_values(train_data, train_dialogue_id)
    val_dialogue_name, val_dialogue_length, val_dialogue_text, val_dialogue_label= get_values(val_data, val_dialogue_id)
    test_dialogue_name, test_dialogue_length, test_dialogue_text, test_dialogue_label= get_values(test_data, test_dialogue_id)


    assert(train_dialogue_label.shape[0] == train_dialogue_text.shape[0])
    assert(val_dialogue_label.shape[0] == val_dialogue_text.shape[0])
    assert(test_dialogue_label.shape[0] == test_dialogue_text.shape[0])

    return train_dialogue_text, val_dialogue_text, test_dialogue_text, train_dialogue_label, \
           val_dialogue_label, test_dialogue_label, train_dialogue_length, val_dialogue_length, test_dialogue_length, \
           max_utts, train_dialogue_name, val_dialogue_name, test_dialogue_name


def load_text_data():

    train_dialogue_text, val_dialogue_text, test_dialogue_text, train_dialogue_label, \
        val_dialogue_label, test_dialogue_label, train_dialogue_length, val_dialogue_length, test_dialogue_length, max_utts, \
        train_dialogue_name, val_dialogue_name, test_dialogue_name = \
        get_dialogue_text_ids(train_dialogue_ids, val_dialogue_ids, test_dialogue_ids, train_data, val_data, test_data)

    # Creating masks
    train_mask, val_mask, test_mask = get_masks(train_dialogue_length, val_dialogue_length, test_dialogue_length)

    assert(train_dialogue_name==list(train_dialogue_ids.keys()))
    assert(val_dialogue_name==list(val_dialogue_ids.keys()))
    assert(test_dialogue_name==list(test_dialogue_ids.keys()))

    return train_dialogue_text, val_dialogue_text, test_dialogue_text, train_dialogue_label, \
           val_dialogue_label, test_dialogue_label, train_mask, val_mask, test_mask, W, train_dialogue_name, val_dialogue_name, test_dialogue_name, MODE

#########################################################################################################################################################

def prepare_bimodal(ID, text, audio):

    bimodal=[]
    for vid, utts in ID.items():
        bimodal.append(np.concatenate( (text[vid],audio[vid]) , axis=1))
        # bimodal.append(np.asarray(text[vid]))
    return np.asarray(bimodal)

def load_bimodal_data():

    TEXT_UNIMODAL = "./output/text_unimodal_{}.pkl".format(MODE.lower())
    AUDIO_UNIMODAL = "./output/audio_unimodal_{}.pkl".format(MODE.lower())
    #Load features
    train_text_x, val_text_x, test_text_x = pickle.load(open(TEXT_UNIMODAL, "rb"), encoding='latin1')
    train_audio_x, val_audio_x, test_audio_x = pickle.load(open(AUDIO_UNIMODAL, "rb"), encoding='latin1')

    train_bimodal_x = prepare_bimodal(train_dialogue_ids, train_text_x, train_audio_x)
    val_bimodal_x = prepare_bimodal(val_dialogue_ids, val_text_x, val_audio_x)
    test_bimodal_x = prepare_bimodal(test_dialogue_ids, test_text_x, test_audio_x)

    train_dialogue_length, val_dialogue_length, test_dialogue_length=get_dialogue_lengths(train_dialogue_ids, \
                                val_dialogue_ids, test_dialogue_ids)

    train_dialogue_labels, val_dialogue_labels, test_dialogue_labels=get_dialogue_labels(train_dialogue_ids, \
                                val_dialogue_ids, test_dialogue_ids)

    train_mask, val_mask, test_mask = get_masks(train_dialogue_length, val_dialogue_length, test_dialogue_length)

    return train_bimodal_x, val_bimodal_x, test_bimodal_x, train_dialogue_labels, \
           val_dialogue_labels, test_dialogue_labels, train_mask, val_mask, test_mask, \
           train_dialogue_ids.keys(), val_dialogue_ids.keys(), test_dialogue_ids.keys(), MODE


