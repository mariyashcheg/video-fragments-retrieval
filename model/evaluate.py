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

def evaluate(model, video_iterator, lang_iterator, annotations, device, preliminary=100, 
                model_types=['model'], iou_thresholds=[0.5,0.7]):

    videos = {}
    # compute embeddings for all clips in each unique video in dataset
    for batch in tqdm(video_iterator):
        with torch.no_grad():
            videos[batch['video']] = model(batch['feature'].to(device))

    recalls = {(model_type, iou_thr): {1: [], 10: [], 100: [], 'MR':[]} 
        for (model_type, iou_thr) in itertools.product(model_types, iou_thresholds)}
    moments = lang_iterator.batch_sampler.moments
    print('\nEvaluation:')
    # compute embeddings for all queries in dataset
    for li, batch in tqdm(enumerate(lang_iterator)):
        with torch.no_grad():
            lang_emb = model(batch['feature'].to(device), False, device)

        # ground truth: 1 if IoU between moment and at least 2 manual annotations is greater the threshold
        # for incorrect videos all IoU score are 0
        distances, predicts = [], {key: [] for key in recalls.keys()}
        for video in videos.keys():
            video_emb = videos[video]
            num_segments = video_emb.size(0)
            # for each query compute distance between query and clip embeddings
            dist = F.pairwise_distance(video_emb, lang_emb.repeat(num_segments, 1))
            # for each possible moment in video compute distance between moment and query
            # as sum of distances between clips in the moment and query (precomputed on prev step)
            # (15 possible moments for video of 5 clips, 21 - for 6 clips)
            for start_t, end_t in moments[num_segments]:
                distances.append(dist.index_select(0, torch.arange(start_t, end_t+1).to(device)).mean().item())
                if video == batch['video']:
                    ious = utils.get_iou(annotations[batch['annot_id']]['times'], start_t, end_t)
                    for iou_thr in iou_thresholds:
                        predicts[('model', iou_thr)].append(int(( ious > iou_thr ).sum() >= 2))
                else:
                    for iou_thr in iou_thresholds:
                        predicts[('model', iou_thr)].append(0)

        # sort all moments from all videos by distances
        ind_rand = np.random.choice(np.arange(len(distances)), size=len(distances), replace=False)
        for iou_thr in iou_thresholds:
            ground_truth = np.array(predicts[('model',iou_thr)], dtype=int)
            predicts[('model', iou_thr)] = ground_truth[np.argsort(distances)]
            predicts[('chance',iou_thr)] = ground_truth[ind_rand]
            for model_type in model_types:
                for k in recalls[(model_type, iou_thr)].keys():
                    if k == 'MR':
                        # median rank: index position of first correct moment
                        recalls[(model_type, iou_thr)][k].append(np.where(predicts[(model_type, iou_thr)] == 1)[0][0])
                    else:
                        # Recall@K: 1 if correct moment is among in top-K moments, 0 otherwise
                        recalls[(model_type, iou_thr)][k].append(int(predicts[(model_type, iou_thr)][:k].sum() > 0))

        if (li % preliminary == 0) and (li != 0):
            print()
            for (model_type, iou_thr) in recalls.keys():
                print(f'{model_type}, IoU={iou_thr}:\t', ''.join([f'{name}: {value:.4f}\t' 
                    for name, value in get_metrics(recalls[(model_type, iou_thr)]).items()]))
            print()

    return {f'{model_type}, IoU={iou_thr}': get_metrics(recalls[(model_type, iou_thr)]) 
                            for (model_type, iou_thr) in recalls.keys()}


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
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--model_types", type=str, nargs='+', default=[])
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
    MODEL_TYPES = [model_type for model_type in args.model_types
                            if model_type in ['model','chance']]
    if MODEL_TYPES == []:
        MODEL_TYPES = ['model']
    print(DEVICE, FEATURE_TYPE, DATASET_TYPE, args.iou, MODEL_TYPES)

    moment_frequency = {}
    if 'prior' in MODEL_TYPES:
        
        print('Loading train dataset:')
        train_annotations, train_videos = utils.load_dataset_info('train', DATASET_DIRECTORY, args.missed_videos)
        train_dataset = data.CustomDataset(train_videos, train_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE)

        for _, annot_info in train_annotations.items():
            num_segments = train_dataset.num_segments_info[annot_info['video']]
            if num_segments not in moment_frequency.keys():
                moment_frequency[num_segments] = {tuple(moment): 0 for moment in utils.generate_moments(num_segments)}
                moment_frequency[num_segments]['total'] = 0
            else:
                for t in annot_info['times']:
                    moment_frequency[num_segments][tuple(t)] += 1
                    moment_frequency[num_segments]['total'] += 1
                    
        for num_segments in moment_frequency.keys():
            moment_frequency[num_segments] = [freq / moment_frequency[num_segments]['total'] 
                        for moment, freq in moment_frequency[num_segments].items() if moment != 'total']

    print('Loading validation dataset:')
    val_annotations, val_videos = utils.load_dataset_info(DATASET_TYPE, DATASET_DIRECTORY, args.missed_videos)

    val_dataset = data.CustomDataset(val_videos, val_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE, validate=True)
    val_video_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.VideoBatchSampler(val_videos, val_dataset.num_segments_info))
    val_lang_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.LanguageBatchSampler(val_annotations, val_dataset.num_segments_info, args.iou))
    
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
        model, val_video_iter, val_lang_iter, val_annotations, DEVICE, args.preliminary_print, MODEL_TYPES)

    for (model_type, iou_thr) in metrics.keys():
        print(f'{model_type}, IoU={iou_thr}:\t',''.join([f'{name}: {value:.4f}\t' 
            for name, value in metrics[(model_type, iou_thr)].items()]))

    with open(Path(EXPER_DIRECTORY).joinpath('metrics.json'), 'w') as f:
        json.dump(metrics, f)
