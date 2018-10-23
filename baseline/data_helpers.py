import numpy as np
import pandas as pd
import pickle
import os, sys
from collections import Counter, defaultdict

###################################################################################################################################

# Hyperparams
max_length=50 # Maximum length of the sentence

class Dataloader:
    
    def __init__(self, mode=None):

        try:
            assert(mode is not None)
        except AssertionError as e:
            print("Set mode as 'Sentiment' or 'Emotion'")
            exit()

        self.MODE = mode # Sentiment or Emotion classification mode
        self.max_l = max_length

        """
            Loading the dataset: 
                - revs is a dictionary with keys/value: 
                    - text: original sentence
                    - split: train/val/test :: denotes the which split the tuple belongs to
                    - y: label of the sentence
                    - dialog: ID of the dialog the utterance belongs to
                    - utterance: utterance number of the dialog ID
                    - num_words: number of words in the utterance
                - W: glove embedding matrix
                - vocab: the vocabulary of the dataset
                - word_idx_map: mapping of each word from vocab to its index in W
                - label_index: mapping of each label (emotion or sentiment) to its assigned index, eg. label_index['neutral']=0
        """
        x = pickle.load(open("./data/pickles/data_{}.p".format(self.MODE.lower()),"rb"))
        revs, self.W, self.word_idx_map, self.vocab, _, label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        self.num_classes = len(label_index)
        print("Labels used for this classification: ", label_index)


        # Preparing data
        self.train_data, self.val_data, self.test_data = {},{},{}
        for i in range(len(revs)):
            
            utterance_id = revs[i]['dialog']+"_"+revs[i]['utterance']
            sentence_word_indices = self.get_word_indices(revs[i]['text'])
            label = label_index[revs[i]['y']]

            if revs[i]['split']=="train":
                self.train_data[utterance_id]=(sentence_word_indices,label)
            elif revs[i]['split']=="val":
                self.val_data[utterance_id]=(sentence_word_indices,label)
            elif revs[i]['split']=="test":
                self.test_data[utterance_id]=(sentence_word_indices,label)


        # Creating dialogue:[utterance_1, utterance_2, ...] ids
        self.train_dialogue_ids = self.get_dialogue_ids(self.train_data.keys())
        self.val_dialogue_ids = self.get_dialogue_ids(self.val_data.keys())
        self.test_dialogue_ids = self.get_dialogue_ids(self.test_data.keys())

        # Max utternance in a dialog in the dataset
        self.max_utts = self.get_max_utts(self.train_dialogue_ids, self.val_dialogue_ids, self.test_dialogue_ids)


    def get_word_indices(self, data_x):
        length = len(data_x.split())
        return np.array([self.word_idx_map[word] for word in data_x.split()] + [0]*(self.max_l-length))[:self.max_l]

    def get_dialogue_ids(self, keys):
        ids=defaultdict(list)
        for key in keys:
            ids[key.split("_")[0]].append(int(key.split("_")[1]))
        for ID, utts in ids.items():
            ids[ID]=[str(utt) for utt in sorted(utts)]
        return ids

    def get_max_utts(self, train_ids, val_ids, test_ids):
        max_utts_train = max([len(train_ids[vid]) for vid in train_ids.keys()])
        max_utts_val = max([len(val_ids[vid]) for vid in val_ids.keys()])
        max_utts_test = max([len(test_ids[vid]) for vid in test_ids.keys()])
        return np.max([max_utts_train, max_utts_val, max_utts_test])

    def get_one_hot(self, label):
        label_arr = [0]*self.num_classes
        label_arr[label]=1
        return label_arr[:]


    def get_dialogue_audio_embs(self):
        key = list(self.train_audio_emb.keys())[0]
        pad = [0]*len(self.train_audio_emb[key])

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
                for _ in range(self.max_utts-len(local_audio)):
                    local_audio.append(pad[:])
                dialogue_audio.append(local_audio[:self.max_utts])
            return np.array(dialogue_audio)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_audio_emb)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_audio_emb)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_audio_emb)

    def get_dialogue_text_embs(self):
        key = list(self.train_data.keys())[0]
        pad = [0]*len(self.train_data[key][0])

        def get_emb(dialogue_id, local_data):
            dialogue_text = []
            for vid in dialogue_id.keys():
                local_text = []
                for utt in dialogue_id[vid]:
                    local_text.append(local_data[vid+"_"+str(utt)][0][:])
                for _ in range(self.max_utts-len(local_text)):
                    local_text.append(pad[:])
                dialogue_text.append(local_text[:self.max_utts])
            return np.array(dialogue_text)

        self.train_dialogue_features = get_emb(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_features = get_emb(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_features = get_emb(self.test_dialogue_ids, self.test_data)


    def get_dialogue_labels(self):

        def get_labels(ids, data):
            dialogue_label=[]

            for vid, utts in ids.items():
                local_labels=[]
                for utt in utts:
                    local_labels.append(self.get_one_hot(data[vid+"_"+str(utt)][1]))
                for _ in range(self.max_utts-len(local_labels)):
                    local_labels.append(self.get_one_hot(1)) # Dummy label
                dialogue_label.append(local_labels[:self.max_utts])
            return np.array(dialogue_label)

        self.train_dialogue_label=get_labels(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_label=get_labels(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_label=get_labels(self.test_dialogue_ids, self.test_data)

    def get_dialogue_lengths(self):

        self.train_dialogue_length, self.val_dialogue_length, self.test_dialogue_length=[], [], []
        for vid, utts in self.train_dialogue_ids.items():
            self.train_dialogue_length.append(len(utts))
        for vid, utts in self.val_dialogue_ids.items():
            self.val_dialogue_length.append(len(utts))
        for vid, utts in self.test_dialogue_ids.items():
            self.test_dialogue_length.append(len(utts))

    def get_masks(self):

        self.train_mask = np.zeros((len(self.train_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.train_dialogue_length)):
            self.train_mask[i,:self.train_dialogue_length[i]]=1.0
        self.val_mask = np.zeros((len(self.val_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.val_dialogue_length)):
            self.val_mask[i,:self.val_dialogue_length[i]]=1.0
        self.test_mask = np.zeros((len(self.test_dialogue_length), self.max_utts), dtype='float')
        for i in range(len(self.test_dialogue_length)):
            self.test_mask[i,:self.test_dialogue_length[i]]=1.0


    def load_audio_data(self, ):

        AUDIO_PATH = "./data/pickles/audio_embeddings_feature_selection_{}.pkl".format(self.MODE.lower())
        self.train_audio_emb, self.val_audio_emb, self.test_audio_emb = pickle.load(open(AUDIO_PATH,"rb"))
        
        self.get_dialogue_audio_embs()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()

    def load_text_data(self, ):

        self.get_dialogue_text_embs()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()


    def load_bimodal_data(self,):
        
        TEXT_UNIMODAL = "./data/pickles/text_{}.pkl".format(self.MODE.lower())
        AUDIO_UNIMODAL = "./data/pickles/audio_{}.pkl".format(self.MODE.lower())

        #Load features
        train_text_x, val_text_x, test_text_x = pickle.load(open(TEXT_UNIMODAL, "rb"), encoding='latin1')
        train_audio_x, val_audio_x, test_audio_x = pickle.load(open(AUDIO_UNIMODAL, "rb"), encoding='latin1')

        def concatenate_fusion(ID, text, audio):
            bimodal=[]
            for vid, utts in ID.items():
                bimodal.append(np.concatenate( (text[vid],audio[vid]) , axis=1))
            return np.array(bimodal)

        self.train_dialogue_features = concatenate_fusion(self.train_dialogue_ids, train_text_x, train_audio_x)
        self.val_dialogue_features = concatenate_fusion(self.val_dialogue_ids, val_text_x, val_audio_x)
        self.test_dialogue_features = concatenate_fusion(self.test_dialogue_ids, test_text_x, test_audio_x)

        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()


