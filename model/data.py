import string
import random
import re
import numpy as np
import torch
import torch.nn as nn
import itertools
import h5py

from utils import generate_moments, get_iou

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict, Counter
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
POOLING = dict(
    avg=np.mean,
    max=np.max)
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
        
        self.word_embeddings = nn.Embedding.from_pretrained(
                embeddings=self.get_embeddings(), freeze=True, padding_idx=0)
        
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
        idx_tensor, lens_tensor = self.idx2tensor(idx, align, word_len=tensor_size) # MAX_SEN_LEN
        word_tensor = self.word_embeddings(idx_tensor)
        return word_tensor, lens_tensor

    def idx2tensor(self, idx_sequences, align='left', word_len=-1):
        batch_size = len(idx_sequences)
        if word_len == -1:
            word_len = max([len(idx_seq) for idx_seq in idx_sequences])
        tensor = torch.zeros(batch_size, word_len, dtype=torch.long)
        lens = torch.zeros(batch_size, dtype=torch.long)

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
            lens[k] = curr_seq_len

        return tensor, lens
    
    def get_embeddings(self):
        emb_matrix = torch.zeros(self.get_items_count(), self.emb_dim)
        for word, idx in self.item2idx_dict.items():
            emb_matrix[idx] = torch.tensor(self.embedding_dict[word])
        
        return emb_matrix


class CustomDataset(Dataset):

    def __init__(self, videos, annotations, ft_directory, ft_type, word_indexer, validate=False, max_query_len=20):
        self.word_indexer = word_indexer
        self.max_query_len = max_query_len
        self.ft_directory = ft_directory
        self.ft_type = ft_type
        self.num_segments_info = {}
        self.annot_len_info = defaultdict(list)
        self.validate = validate
        # unique video
        self.video_features = {}
        self.load_video_features(videos)
        # unique lang queries
        self.lang_features = {}
        self.lang_len_seq = {}
        self.annotations = self.load_lang_features(annotations)

    def load_video_features(self, videos):
        for video in tqdm(videos):

            ft_file = f'features_{self.ft_type}/{self.ft_type}_ft_{video}.npy'
            video_features = np.load(Path(self.ft_directory).joinpath(ft_file))
            video_features = video_features.reshape((video_features.shape[0], FEATURE_DIM[self.ft_type]))
            num_segments = video_features.shape[0] // SELECT_FPS

            self.video_features[video] = dict(
                features=torch.from_numpy(video_features[:num_segments*SELECT_FPS, :]).float(),
                num_segments=num_segments
            )
            self.num_segments_info[video] = num_segments
            
    def load_lang_features(self, annotations):
        annotations_correct = {}        
        for annot_id, annot_info in tqdm(annotations.items()):
            num_segments = self.num_segments_info[annot_info['video']]
            times = np.array(annot_info['times'])
            times = times[times[:, 1] < num_segments]
            times = times[np.where((times[:, 1] - times[:, 0] + 1) < num_segments)[0]]
            correct = max([int(( get_iou(times, start_t, end_t) >= 0.7 ).sum() >= 2) 
                                for start_t, end_t in generate_moments(num_segments)])
            if correct > 0:
                new_annot_info = deepcopy(annot_info)
                new_annot_info['times'] = times.tolist()
                annotations_correct[annot_id] = new_annot_info
                for t in set(times[:, 1] - times[:, 0]+1):
                    self.annot_len_info[t].append(annot_id)
                query = annot_info['description'].rstrip('\n ').lower()
                words = [i[1] for i in re.findall("('\w )|([\w\d]+)", query) if i[1] != '']
                # words = re.sub(f'(?=[^\s])(?=[{string.punctuation}])', ' ', query).split(' ')
                features, lens = self.word_indexer.items2tensor([words], self.max_query_len)
                self.lang_features[annot_id] = features
                self.lang_len_seq[annot_id] = lens

        for k,v in self.annot_len_info.items():
            self.annot_len_info[k] = set(v) 
        print(len(annotations_correct), '/', len(annotations))
        return annotations_correct   
        
    def __getitem__(self, sample):
        if self.validate:
            if 'annotation_id' in sample.keys():
                # language
                sample_features = dict(
                    features=self.lang_features[sample['annotation_id']],
                    len_seq=self.lang_len_seq[sample['annotation_id']],
                    video=sample['video_pos'],
                    annot_id=sample['annotation_id']
                )
            else:
                # video 
                sample_features = dict(
                    features=self.video_features[sample['video_pos']]['features'],
                    video=sample['video_pos'],
                    endpoints=tuple((sample['start_t'], sample['end_t']))
                )
        else:
            sample_features = dict(
                posit=self.video_features[sample['video_pos']]['features'],
                negat=self.video_features[sample['video_neg']]['features'], 
                lang=self.lang_features[sample['annotation_id']],
                len_seq=self.lang_len_seq[sample['annotation_id']],
                endp_posit=tuple((sample['start_t'], sample['end_t'])),
                endp_negat=tuple((sample['start_tn'], sample['end_tn']))
            )
        return sample_features


class CustomBatchSampler(BatchSampler):

    def __init__(self, batch_size, annotations, num_segments_info, annot_len_info, 
            train=True, drop_last=False, same_length=True):
        self.batch_size = batch_size
        self.annotations = annotations
        self.num_segments_info = num_segments_info
        self.annot_len_info = annot_len_info
        self.train = train
        self.drop_last = drop_last
        self.same_length = same_length
        self.indices = defaultdict(list)
        for video, num_segments in self.num_segments_info.items():
            self.indices[num_segments].append(video)
        
    def get_annotations(self, annot_id, len_annot=-1):
        annot_info = self.annotations[annot_id]
        annotations = np.array(annot_info['times'])
        annotations = annotations[annotations[:, 1] < self.num_segments_info[annot_info['video']]]
        if len_annot > 0:
            annotations = annotations[np.where((annotations[:, 1] - annotations[:, 0] + 1) == len_annot)[0]]
        return annotations

    def get_possible_videos(self, num_segments):
        possible_videos = []
        for num in self.indices.keys():
            if num_segments <= num:
                possible_videos.extend(self.indices[num])
        return possible_videos
            
    def __iter__(self):
        # annotation_ids = list(self.annotations.keys())
        lens_annot_ids = deepcopy(self.annot_len_info)
        batch, annotation_ids = [], []
        for k, v in lens_annot_ids.items():
            annotation_ids.extend([(k,i) for i in v if len(v) >= self.batch_size])
        # print(len(annotation_ids)) 

        while len(annotation_ids) >= self.batch_size:
            len_annot, annot_id = random.choice(annotation_ids)
            for k in lens_annot_ids.keys():
                lens_annot_ids[k].discard(annot_id)

            for annot_id in [annot_id]+random.sample(lens_annot_ids[len_annot], k=self.batch_size-1):
                start_t, end_t = random.choice(self.get_annotations(annot_id, len_annot))
                # lens_annot_ids[len_annot].discard(annot_id)
                for k in lens_annot_ids.keys():
                    lens_annot_ids[k].discard(annot_id)

                annot_info = self.annotations[annot_id]
                num_segments = self.num_segments_info[annot_info['video']]
                # positive sample
                
                # intra-negative sample (wrong segments from the same video):
                #    select segment of the same length (???) in the video which IoU with posit is less then 1
                #    which means that segments differ at least at one clip
                possible_segments = generate_moments(num_segments)
                possible_segments = [segment for segment in possible_segments 
                    if get_iou([(start_t, end_t)], segment[0], segment[1]) < 1]
                        # if (segment[1] - segment[0]) == (end_t - start_t) and (segment != (start_t, end_t))]
                
                start_tn, end_tn = random.choice(possible_segments)  
                # inter-negative sample:
                #    select the same segment (start_t, end_t) from random video in dataset
                possible_neg_videos = self.get_possible_videos(end_t+1)
                possible_neg_videos.remove(annot_info['video'])
                if self.train:
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

            yield batch
            batch, annotation_ids = [], []
            for k, v in lens_annot_ids.items():
                annotation_ids.extend([(k,i) for i in v if len(v) >= self.batch_size])
            
            # if len(batch) == self.batch_size:
            #     yield batch
            #     batch, annotation_ids = [], []
            #     for k, v in lens_annot_ids.items():
            #         annotation_ids.extend([(k,i) for i in v if len(v) >= self.batch_size])


def custom_collate(batch):
    posit, negat, lang, len_seq = [], [], [], []
    endp_posit, endp_negat = [], []
    for i, sample in enumerate(batch):
        posit.append(sample['posit'])
        negat.append(sample['negat'])
        lang.append(sample['lang'])
        len_seq.append(sample['len_seq'])
        endp_posit.append(sample['endp_posit'])
        endp_negat.append(sample['endp_negat'])
    return dict(
        posit=posit, #torch.cat(posit, axis=0),
        negat=negat, #torch.cat(negat, axis=0),  
        lang=torch.cat(lang, axis=0),
        len_seq=torch.cat(len_seq, axis=0),
        endp_posit=torch.LongTensor(endp_posit),
        endp_negat=torch.LongTensor(endp_negat)
    )


class VideoBatchSampler(BatchSampler):

    def __init__(self, videos, num_segments_info):
        self.videos = videos
        self.num_segments_info = num_segments_info
                
    def __iter__(self):
        batch = []
        for video in self.videos:
            batch.append(
                dict(
                    video_pos=video,
                    start_t=0,
                    end_t=self.num_segments_info[video] - 1,
                )
            )
            yield batch
            batch = []
            
    def __len__(self):
        return len(self.videos)    

class LanguageBatchSampler(BatchSampler):

    def __init__(self, annotations, num_segments_info):
        self.annotations = annotations
        self.num_segments_info = num_segments_info
        self.moments = {num_seg: generate_moments(num_seg) for num_seg in range(7)}
        
    def get_annotations(self, annot_id):
        annot_info = self.annotations[annot_id]
        annotations = np.array(annot_info['times'])
        annotations = annotations[annotations[:, 1] <= self.num_segments_info[annot_info['video']]]
        return annotations
    
    def __iter__(self):
        batch = []
        for annot_id in list(self.annotations.keys()):
            annot_info = self.annotations[annot_id]
            num_segments = self.num_segments_info[annot_info['video']]
            batch.append(
                dict(
                    annotation_id=annot_id,
                    video_pos=annot_info['video'],
                )
            )
            yield batch
            batch = []
            
    def __len__(self):
        return len(self.annotations)


def validate_collate(batch):
    annot_id = batch[0]['annot_id'] if 'annot_id' in batch[0].keys() else []
    len_seq = batch[0]['len_seq'] if 'len_seq' in batch[0].keys() else []
    endpoints = batch[0]['endpoints'] if 'endpoints' in batch[0].keys() else []
    return dict(
        feature=batch[0]['features'],
        video=batch[0]['video'],
        annot_id=annot_id,
        len_seq=len_seq,
        endpoints=torch.LongTensor(endpoints)
    )