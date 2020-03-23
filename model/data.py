import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
import itertools
from collections import defaultdict

from tqdm import tqdm
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler

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

    def __init__(self, videos, annotations, word_indexer, ft_directory, ft_type, max_query_len=50):
        self.word_indexer = word_indexer
        self.max_query_len = max_query_len
        self.ft_directory = ft_directory
        self.ft_type = ft_type
        self.num_segments_info = {}
        # unique video
        self.video_features = {}
        self.load_video_features(videos)
        # unique lang queries
        self.lang_features = {}
        self.load_lang_features(annotations)

    def load_video_features(self, videos):
        for video in tqdm(videos):
            ft_file = '{}_ft_{}.npy'.format(self.ft_type, video)
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
            
            self.video_features[video] = dict(
                segment_features=segment_features,
                context_features=context_features,
                num_segments=num_segments
            )
            self.num_segments_info[video] = num_segments
            
    def load_lang_features(self, annotations):        
        for annot_id, annot_info in tqdm(annotations.items()):
            query = annot_info['description'].rstrip('\n ').lower()
            words = re.sub(f'(?=[^\s])(?=[{string.punctuation}])', ' ', query).split(' ')
            self.lang_features[annot_id] = self.word_indexer.items2tensor(
                [words], self.max_query_len) 
            
    def make_visual_features(self, video, start_t, end_t):
        num_segments = self.video_features[video]['num_segments']
        segment_ft = torch.from_numpy(self.video_features[video]['segment_features'][start_t:end_t+1]).float()
        context_ft = torch.from_numpy(self.video_features[video]['context_features'].reshape(1, -1)).float()
        temporal_endpoints = torch.cat(
            [torch.arange(start_t, end_t+1).view(-1,1),
             torch.arange(start_t+1, end_t+2).view(-1,1)], axis=1).float() / num_segments
        features = torch.cat(
            [segment_ft, context_ft.repeat(segment_ft.size(0), 1), temporal_endpoints], axis=1)
        return features    
        
    def __getitem__(self, sample):
        # positive 
        posit_features = self.make_visual_features(sample['video_pos'], sample['start_t'], sample['end_t'])
        # intra-negative
        intra_features = self.make_visual_features(sample['video_pos'], sample['start_tn'], sample['end_tn'])      
        # inter-negative        
        inter_features = self.make_visual_features(sample['video_neg'], sample['start_t'], sample['end_t'])        
        
        # lang features
        lang_features = self.lang_features[sample['annotation_id']]
        
        return dict(
            posit=posit_features,
            intra=intra_features, 
            inter=inter_features, 
            lang=lang_features)


class CustomBatchSampler(BatchSampler):

    def __init__(self, batch_size, annotations, num_segments_info, shuffle=True, drop_last=False):
        self.batch_size = batch_size
        self.annotations = annotations
        self.num_segments_info = num_segments_info
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = defaultdict(list)
        for video, num_segments in self.num_segments_info.items():
            self.indices[num_segments].append(video)
        
    def get_annotations(self, annot_id):
        annot_info = self.annotations[annot_id]
        annotations = np.array(annot_info['times'])
        annotations = annotations[annotations[:, 1] <= self.num_segments_info[annot_info['video']]]
        return annotations

    def get_possible_videos(self, num_segments):
        possible_videos = []
        for num in self.indices.keys():
            if num_segments <= num:
                possible_videos.extend(self.indices[num])
        return possible_videos
        
    def __iter__(self):
        annotation_ids = list(self.annotations.keys())
        if self.shuffle:
            random.shuffle(annotation_ids)
        batch = []
        curr_batch_size = 0
        
        for annot_id in annotation_ids:
            annot_info = self.annotations[annot_id]
            num_segments = self.num_segments_info[annot_info['video']]
            # positive sample
            start_t, end_t = random.choice(self.get_annotations(annot_id)) 
            
            # intra-negative sample (wrong segments from the same video):
            #    select segment of the same length (???) in the video which IoU with posit is less then 1
            #    which means that segments differ at least at one clip
            possible_segments = [(j,j) for j in range(num_segments)]
            for j in itertools.combinations(range(num_segments), 2):
                possible_segments.append(j)
            possible_segments = [segment for segment in possible_segments 
                        if (segment[1] - segment[0]) == (end_t - start_t) and (segment != (start_t, end_t))]
            if self.shuffle:
                start_tn, end_tn = random.choice(possible_segments)    
            else:
                start_tn, end_tn = possible_segments[0]
            
            # inter-negative sample:
            #    select the same segment (start_t, end_t) from random video in dataset
            possible_neg_videos = self.get_possible_videos(end_t+1)
            possible_neg_videos.remove(annot_info['video'])
            if self.shuffle:
                video_neg = random.choice(possible_neg_videos)
            else:
                video_neg = possible_neg_videos[0]
            
            batch.append(dict(
                annotation_id=annot_id,
                video_pos=annot_info['video'],
                video_neg=video_neg,
                start_t=start_t,
                end_t=end_t,
                start_tn=start_tn,
                end_tn=end_tn
            ))
            curr_batch_size += (end_t - start_t + 1)
            
            if int(curr_batch_size) >= self.batch_size:
                yield batch
                batch = []
                curr_batch_size = 0
                
        if len(batch) > 0 and not self.drop_last:
            yield batch
            
    def __len__(self):
        if self.drop_last:
            return len(self.annotations) // self.batch_size
        else:
            return (len(self.annotations) + self.batch_size - 1) // self.batch_size


def custom_collate(batch):
    mask = []
    posit, intra, inter, lang = [], [], [], []
    for i, sample in enumerate(batch):
        posit.append(sample['posit'])
        intra.append(sample['intra'])
        inter.append(sample['inter'])
        lang.append(sample['lang'])
        mask.extend([i]*sample['posit'].size(0))
    return dict(
        posit=torch.cat(posit, axis=0),
        intra=torch.cat(intra, axis=0), 
        inter=torch.cat(inter, axis=0), 
        lang=torch.cat(lang, axis=0),
        mask=torch.LongTensor(mask)
    )