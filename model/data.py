import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from tqdm import tqdm
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

SELECT_FPS = 25
FRAMES_PER_SEC = 5
SEC_PER_SEGMENT = 5
FEATURE_DIM = dict(
    vgg19=4096,
    resnet152=2048)
EMBEDDING_DIM = 100
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class WordIndexer():
    def __init__(self, emb_path, emb_dim=EMBEDDING_DIM, vocab=None):
        self.pad = PAD_TOKEN
        self.unk = UNK_TOKEN
        self.emb_dim = emb_dim
        
        self.item2idx_dict = dict()
        self.idx2item_dict = dict()
        self.embedding_dict = dict()
        self.add_item(self.pad, [0]*self.emb_dim)
        
        with open(Path(emb_path).joinpath(f'glove.6B.{self.emb_dim}d.txt'), 'r', encoding='UTF-8') as file:
            lines = file.readlines()
            for line in tqdm(lines):
                row = line.strip().split(' ')
                vector = [float(i) for i in row[1:]]
                assert len(vector) == self.emb_dim
                if vocab is None:
                    self.add_item(row[0], vector)
                elif row[0] in vocab:
                    self.add_item(row[0], vector)
        
    def get_items_list(self):
        return list(self.item2idx_dict.keys())

    def get_items_count(self):
        return len(self.item2idx_dict.keys())

    def add_item(self, item, item_vector):
        idx = self.get_items_count()
        self.item2idx_dict[item] = idx
        self.idx2item_dict[idx] = item
        self.embedding_dict[item] = item_vector
        return idx

    def items2idx(self, item_sequences):
        idx_sequences = []
        for item_seq in item_sequences:
            idx_seq = list()
            for item in item_seq:
                if item in self.item2idx_dict:
                    idx_seq.append(self.item2idx_dict[item])
                else:
                    idx_seq.append(self.item2idx_dict[self.unk])
            idx_sequences.append(idx_seq)
        return idx_sequences

    def idx2items(self, idx_sequences):
        item_sequences = []
        for idx_seq in idx_sequences:
            item_seq = [self.idx2item_dict[idx] for idx in idx_seq]
            item_sequences.append(item_seq)
        return item_sequences

    def items2tensor(self, item_sequences, tensor_size, align='left'):
        idx = self.items2idx(item_sequences)
        word_tensor = self.idx2tensor(idx, align, word_len=tensor_size) # MAX_SEN_LEN
        return word_tensor

    def idx2tensor(self, idx_sequences, align='left', word_len=-1):
        batch_size = len(idx_sequences)
        if word_len == -1:
            word_len = max([len(idx_seq) for idx_seq in idx_sequences])
        tensor = torch.zeros(batch_size, word_len, dtype=torch.long)

        for k, idx_seq in enumerate(idx_sequences):
            curr_seq_len = len(idx_seq)
            if curr_seq_len > word_len:
                idx_seq = [idx_seq[i] for i in range(word_len)]
                curr_seq_len = word_len
            if align == 'left':
                tensor[k, :curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            elif align == 'center':
                start_idx = (word_len - curr_seq_len) // 2
                tensor[k, start_idx:start_idx+curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            else:
                raise ValueError('Unknown align string.')

        return tensor
    
    def get_embeddings(self):
        emb_matrix = torch.zeros(self.get_items_count(), self.emb_dim)
        for word, idx in self.item2idx_dict.items():
            emb_matrix[idx] = torch.tensor(self.embedding_dict[word])
        
        return emb_matrix


class CustomDataset(Dataset):

    def __init__(self, video_list, word_indexer, ft_directory, ft_type, max_query_len=50):
        self.word_indexer = word_indexer
        self.max_query_len = max_query_len
        self.ft_directory = ft_directory
        self.ft_type = ft_type
        self.indices = defaultdict(list)
        self.video_list = self.load_features(video_list)

    def load_features(self, video_list):
        video_list_corr = []

        for ind in range(len(video_list)):
            video_info = video_list[ind]
            ft_file = '{}_ft_{}.npy'.format(self.ft_type, video_info['video'])
            video_features = np.load(Path(self.ft_directory).joinpath(ft_file))
            if video_features.shape[0] % SELECT_FPS == 0:
                num_segments = video_features.shape[0] // SELECT_FPS
            else:
                num_segments = video_features.shape[0] // SELECT_FPS + 1

            # segment features = average features for all frames in segment
            segment_features = np.zeros((num_segments, FEATURE_DIM[self.ft_type]))
            for i in range(segment_features.shape[0]):
                segment_features[i, :] = np.mean(video_features[i*FRAMES_PER_SEC:(i+1)*FRAMES_PER_SEC, :], axis=0)
                
            # context features = average features for all frames in video
            context_features = np.mean(video_features, axis=0)

            video_info['segment_features'] = segment_features
            video_info['context_features'] = context_features
            video_info['num_segments'] = num_segments
            self.indices[num_segments].append(ind)
            video_list_corr.append(video_info)

        return video_list_corr
    
    def make_visual_features(self, video_info, start_t, end_t):
        
        num_segments = video_info['num_segments']
        segment_ft = torch.from_numpy(video_info['segment_features'][start_t:end_t+1]).float()
        context_ft = torch.from_numpy(video_info['context_features'].reshape(1, -1)).float()
        temporal_endpoints = torch.cat(
            [torch.arange(start_t, end_t+1).view(-1,1),
             torch.arange(start_t+1, end_t+2).view(-1,1)], axis=1) / num_segments
        
        features = torch.cat(
            [segment_ft, context_ft.repeat(segment_ft.size(0), 1), temporal_endpoints], axis=1)
        return features    
    
    def get_annotations(self, video_info):
        annotations = np.array(video_info['times'])
        annotations = annotations[annotations[:, 1] <= video_info['num_segments']]
        return annotations

    def get_possible_video_indices(self, num_segments):
        possible_indices = []
        for num in self.indices.keys():
            if num_segments <= num:
                possible_indices.extend(self.indices[num])
        return possible_indices
        
    def __getitem__(self, index):
        
        ######### VIDEO FEATURES #########
        video_info = self.video_list[index]
        annotations = self.get_annotations(video_info)
        
        # positive sample
        start_t, end_t = random.choice(annotations)        
        posit_features = self.make_visual_features(video_info, start_t, end_t)
        
        # intra sample (wrong segments from the same video)
        if start_t > video_info['num_segments'] - end_t-1: # take the longest segment
            start_tn, end_tn = 0, start_t-1
            while end_tn - start_tn < end_t - start_t:
                end_tn += 1
        else: 
            start_tn, end_tn = end_t+1, video_info['num_segments']-1
            while end_tn - start_tn < end_t - start_t:
                start_tn -= 1
        intra_features = self.make_visual_features(video_info, start_tn, end_tn)      
        
        # select random video from dataset
        possible_indices = self.get_possible_video_indices(video_info['num_segments'])
        possible_indices.remove(index)
        other_index = random.choice(possible_indices)
        other_video_info = self.video_list[other_index]
        # inter sample (segments from other video)        
        inter_features = self.make_visual_features(other_video_info, start_t, end_t)        
        
        ######### LANGUAGE FEATURES #########
        query = video_info['description'].rstrip('\n ').lower()
        words = re.sub(f'(?=[^\s])(?=[{string.punctuation}])', ' ', query).split(' ')
        lang_features = self.word_indexer.items2tensor([words], self.max_query_len)
        
        return dict(
            posit=posit_features,
            intra=intra_features, 
            inter=inter_features, 
            lang=lang_features)
    
    def __len__(self):
        return len(self.video_list)
