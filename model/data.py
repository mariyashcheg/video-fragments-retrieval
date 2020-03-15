import string
import random
import re
import numpy as np
import torch
import torch.nn as nn

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
        self.video_list = video_list
        self.word_indexer = word_indexer
        self.max_query_len = max_query_len
        self.ft_directory = ft_directory
        self.ft_type = ft_type
        
    def make_average_features(self, video):
        
        ft_file = 'vgg19_ft_{}.npy'.format(video)
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
        
        return segment_features, context_features
    
    
    def make_visual_features(self, start_t, end_t, segment_features, context_features):
        
        num_segments = context_features.shape[0]
        segment_ft = torch.from_numpy(segment_features[start_t:end_t+1]).float()
        context_ft = torch.from_numpy(context_features.reshape(1, -1)).float()
        temporal_endpoints = torch.cat(
            [torch.arange(start_t, end_t+1).view(-1,1),
             torch.arange(start_t+1, end_t+2).view(-1,1)], axis=1) / num_segments
        
        features = torch.cat(
            [segment_ft, context_ft.repeat(segment_ft.size(0), 1), temporal_endpoints], axis=1)
        return features
    
    
    def get_annotations(self, video_info, num_segments):
        annotations = np.array(video_info['times'])
        annotations = annotations[annotations[:, 1] <= num_segments]
        return annotations
    
        
    def __getitem__(self, index):
        
        ######### VIDEO FEATURES #########
        video_info = self.video_list[index]
        segment_features, context_features = self.make_average_features(video_info['video'])
                
        num_segments = segment_features.shape[0]
        annotations = self.get_annotations(video_info, num_segments)
        
        # positive sample
        start_t, end_t = random.choice(annotations)        
        posit_features = self.make_visual_features(start_t, end_t, segment_features, context_features)
        
        # intra sample (wrong segments from the same video)
        if start_t > num_segments - end_t-1: # take the longest segment
            start_tn, end_tn = 0, start_t-1
        elif start_t < num_segments - end_t-1:
            start_tn, end_tn = end_t+1, num_segments-1
        else: # random if both segments of same length
            start_tn, end_tn = random.choice([(0, start_t-1),(end_t+1, num_segments-1)])
        intra_features = self.make_visual_features(start_tn, end_tn, segment_features, context_features)      
        
        # select random video from dataset
        possible_indices = [i for i in range(len(self.video_list)) if i != index]
        other_index = random.choice(possible_indices)
        other_video_info = self.video_list[other_index]
        other_segment_features, other_context_features = self.make_average_features(other_video_info['video'])
        
        other_num_segments = other_segment_features.shape[0]
        other_annotations = self.get_annotations(other_video_info, other_num_segments)
        
        # inter sample (segments from other video)
        start_t, end_t = random.choice(other_annotations)        
        inter_features = self.make_visual_features(start_t, end_t, other_segment_features, other_context_features)        
        
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
