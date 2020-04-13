import argparse
import numpy as np
import json
import re
import random
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
import models
import utils

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

def get_metrics(recalls):
    metrics = {}
    for name, value in recalls.items():
        if name == 'MR':
            metrics[name] = np.median(value)
        else:
            metrics[f'R@{name}'] = np.mean(value)*100
    return metrics

def evaluate(model, video_iterator, lang_iterator, annotations, device, model_types=['model'], prior=[], iou_thresholds=[0.5,0.7]):

    videos = {}
    # compute embeddings for all clips in each unique video in dataset
    for batch in tqdm(video_iterator):
        with torch.no_grad():
            videos[batch['video']] = model(batch['feature'].to(device))

    rank_metrics = {model_type: {1: [], 5: [], 10: [], 'mIoU':[]} for model_type in model_types}
    recall_metrics = {(model_type, thr): {1: [], 5: [], 10: []} 
            for model_type, thr in itertools.product(model_types, iou_thresholds)}

    moments = lang_iterator.batch_sampler.moments
    predicts = {}
    print('\nEvaluation:')
    # compute embeddings for all queries in dataset
    for li, batch in tqdm(enumerate(lang_iterator)):
        with torch.no_grad():
            lang_emb = model(batch['feature'].to(device), False, device)

        video_emb = videos[batch['video']]
        num_segments = video_emb.size(0)

        dist = F.pairwise_distance(video_emb, lang_emb.repeat(num_segments, 1))
        order = np.argsort([dist.index_select(0, torch.arange(start_t, end_t+1).to(device)).mean().item()
                        for start_t, end_t in moments[num_segments]])
        predicts['model'] = [moments[num_segments][i] for i in order][::-1]
        predicts['chance'] = random.sample(moments[num_segments], k=len(moments[num_segments]))
        predicts['prior'] = prior[num_segments]
        
        for model_type in rank_metrics.keys():
            ranks, ious = [], []    
            for time in annotations[batch['annot_id']]['times']:
                ranks.append(predicts[model_type].index(tuple(time))+1)
                ious.append(utils.get_iou([predicts[model_type][0]], time[0], time[1])[0])
            for k in rank_metrics[model_type].keys():
                if k == 'mIoU':
                    rank_metrics[model_type][k].append(np.mean(np.sort(ious)[-3:]))
                else:
                    rank_metrics[model_type][k].append(int(np.mean(np.sort(ranks)[:3]) <= k))
        
        for model_type, iou_thr in recall_metrics.keys():
            ious = np.array([(utils.get_iou(annotations[batch['annot_id']]['times'], start_t, end_t) > iou_thr).sum() >= 2
                        for start_t, end_t in predicts[model_type]]).astype(int)
            for k in recall_metrics[(model_type, iou_thr)].keys():
                recall_metrics[(model_type, iou_thr)][k].append(int(ious[:k].sum() > 0))

    metrics = {}
    for model_type, metrics_dict in rank_metrics.items():
        metrics[model_type] = {}
        for k,v in metrics_dict.items():
            key = k if k == 'mIoU' else f'Rank@{k}'
            metrics[model_type][key] = np.mean(v)*100

    for (model_type, iou_thr), metrics_dict in recall_metrics.items():
        metrics[f'{model_type}, IoU={iou_thr}'] = {}
        for k,v in metrics_dict.items():
            metrics[f'{model_type}, IoU={iou_thr}'][f'Recall@{k}'] = np.mean(v)*100

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
    parser.add_argument("--normalize_lang", type=utils.str2bool, default=False)
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
    MODEL_TYPES = ['model','chance','prior']
    print(DEVICE, FEATURE_TYPE, DATASET_TYPE, MODEL_TYPES)

    moment_frequency = {}
    if 'prior' in MODEL_TYPES:

        print('Loading train dataset:')
        train_annotations, train_videos = utils.load_dataset_info('train', DATASET_DIRECTORY, args.missed_videos)
        train_dataset = data.CustomDataset(train_videos, train_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE)

        for _, annot_info in train_annotations.items():
            num_segments = train_dataset.num_segments_info[annot_info['video']]
            if num_segments not in moment_frequency.keys():
                moment_frequency[num_segments] = {tuple(moment): 0 for moment in utils.generate_moments(num_segments)}
            else:
                for t in annot_info['times']:
                    moment_frequency[num_segments][tuple(t)] += 1
                    
        for num_segments in moment_frequency.keys():
            moment_frequency[num_segments] = sorted(
                moment_frequency[num_segments], key=moment_frequency[num_segments].get, reverse=True)
        
        # moment_frequency = moment_frequency[6]

    print('Loading validation dataset:')
    val_annotations, val_videos = utils.load_dataset_info(DATASET_TYPE, DATASET_DIRECTORY, args.missed_videos)

    val_dataset = data.CustomDataset(val_videos, val_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE, validate=True)
    val_video_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.VideoBatchSampler(val_videos, val_dataset.num_segments_info))
    val_lang_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.LanguageBatchSampler(val_annotations, val_dataset.num_segments_info))
    
    model = models.CALModel(
        pretrained_emb=word_indexer.get_embeddings(),
        visual_input_dim=data.FEATURE_DIM[FEATURE_TYPE]*2+2,
        emb_dim=data.EMBEDDING_DIM, 
        normalize_lang=args.normalize_lang
    )
    model.load_state_dict(state['model_state_dict'])
    model = model.to(DEVICE)    
    model.eval()

    # evaluate model with exhaustive search
    metrics = evaluate(
        model, val_video_iter, val_lang_iter, val_annotations, DEVICE, MODEL_TYPES, moment_frequency)
    for model_type, metrics_dict in metrics.items():
        print(f'{model_type}:\t', ''.join([f'{name}: {value:.2f}\t' for name, value in metrics_dict.items()]))

    with open(Path(EXPER_DIRECTORY).joinpath('metrics_single.json'), 'w') as f:
        json.dump(metrics, f)
