import argparse
import numpy as np
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
import models
import utils

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader


def evaluate(model, video_iterator, lang_iterator, device, preliminary=100):

    videos = {}
    for batch in tqdm(video_iterator):
        with torch.no_grad():
            videos[batch['video']] = model(batch['feature'].to(device))

    recalls = {1: [], 10: [], 100: [], 'MR':[]}
    moments = lang_iterator.batch_sampler.moments
    print('\nEvaluation:')
    for li, batch in tqdm(enumerate(lang_iterator)):
        with torch.no_grad():
            lang_emb = model(batch['feature'].to(device), False, device)

        distances, correct = [],[]
        for video in videos.keys():
            video_emb = videos[video]
            dist = F.pairwise_distance(video_emb, lang_emb.repeat(video_emb.size(0), 1))
            distances.extend([dist.index_select(0, torch.arange(start_t, end_t+1).to(device)).mean().item() 
                                for start_t, end_t in moments[video_emb.size(0)]])
            if video == batch['video']:
                correct.extend(batch['iou'])
            else:
                correct.extend([0]*len(moments[video_emb.size(0)])) #[0]*(mask.max().item()+1))

        correct = np.array(correct, dtype=int)[np.argsort(distances)]
        for k in recalls.keys():
            if k == 'MR':
                recalls[k].append(np.where(correct == 1)[0][0])
            else:
                recalls[k].append(int(correct[:k].sum() > 0))

        if (li % preliminary == 0) and (li != 0):
            metrics = {}
            for name, value in recalls.items():
                if name == 'MR':
                    metrics[name] = np.median(value)
                else:
                    metrics[f'R@{name}'] = np.mean(value)
            print(''.join([f'{name}: {value:.4f}\t' for name, value in metrics.items()]))

    metrics = {}
    for name, value in recalls.items():
        if name == 'MR':
            metrics[name] = np.median(value)
        else:
            metrics[f'R@{name}'] = np.mean(value)
    
    return metrics


if __name__ == '__main__':

    torch.random.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default='features/')
    parser.add_argument("--embedding_dir", type=str, default='glove_pretrained/')
    parser.add_argument("--dataset_dir", type=str, default='dataset/')
    parser.add_argument("--experiment_dir", type=str, default='experiment/')
    parser.add_argument("--missed_videos", type=str, default='../')
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--validation", type=str, default='val')
    parser.add_argument("--preliminary_print", type=int, default=100)
    args = parser.parse_args()

    FT_DIRECTORY = args.features_dir
    EMB_DIRECTORY = args.embedding_dir
    DATASET_DIRECTORY = args.dataset_dir
    EXPER_DIRECTORY = args.experiment_dir
    DATASET_TYPE = args.validation
    
    if args.device in ['cpu','cuda']:
        DEVICE = args.device
    else:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = torch.device('cpu')

    print('Loading word embeddings:')
    word_indexer = data.WordIndexer(EMB_DIRECTORY)

    # load state dict
    state = torch.load(Path(EXPER_DIRECTORY).joinpath('last.pth'))
    FEATURE_TYPE = state['ft_type']
    print(DEVICE, FEATURE_TYPE, DATASET_TYPE)

    print('Loading validation dataset:')
    val_annotations, val_videos = utils.load_dataset_info(DATASET_TYPE, DATASET_DIRECTORY, args.missed_videos)
    val_dataset = data.CustomDataset(val_videos, val_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE, validate=True)
    val_video_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.VideoBatchSampler(val_videos, val_dataset.num_segments_info))
    val_lang_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.LanguageBatchSampler(val_annotations, val_dataset.num_segments_info, 0.5))
    
    model = models.CALModel(
        pretrained_emb=word_indexer.get_embeddings(),
        visual_input_dim=data.FEATURE_DIM[FEATURE_TYPE]*2+2,
        emb_dim=data.EMBEDDING_DIM, 
    )
    model.load_state_dict(state['model_state_dict'])
    model = model.to(DEVICE)    
    
    # evaluate model with exhaustive search
    metrics = evaluate(model, val_video_iter, val_lang_iter, DEVICE, args.preliminary_print)
    print(''.join([f'{name}: {value:.4f}\t' for name, value in metrics.items()]))

