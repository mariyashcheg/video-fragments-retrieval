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


def evaluate(model, iterator, device):

    videos, queries, masks, ious = [], [], [], []
    for batch in tqdm(iterator):
        video_ft = batch['posit'].to(device)
        lang_ft = batch['lang'].to(device)
        masks.append(batch['mask'].to(device))
        ious.append(batch['iou'])

        with torch.no_grad():
            videos.append(model(video_ft))
            queries.append(model(lang_ft, False, device))

    recalls = {1: [], 10: [], 100: [], 'MR':[]}
    for li, lang_emb in tqdm(enumerate(queries)):
        distances = []
        correct = []
        for vi, video_emb in enumerate(videos):
            mask = masks[vi]
            for i in range(mask.max().item()+1):
                mask_i = mask == i
                distances.append(F.pairwise_distance(video_emb[mask_i], lang_emb.repeat(video_emb[mask_i].size(0), 1)).mean().item())
            if vi == li:
                correct.extend(ious[vi])
            else:
                correct.extend([0]*(mask.max().item()+1))
        correct = np.array(correct, dtype=int)[np.argsort(distances)]
        for k in recalls.keys():
            if k == 'MR':
                recalls[k].append(np.where(correct == 1)[0][0])
            else:
                recalls[k].append(int(correct[:k].sum() > 0))

        if (li % 500 == 0) and (li != 0):
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
    args = parser.parse_args()

    FT_DIRECTORY = args.features_dir
    EMB_DIRECTORY = args.embedding_dir
    DATASET_DIRECTORY = args.dataset_dir
    EXPER_DIRECTORY = args.experiment_dir
    
    if args.device in ['cpu','cuda']:
        DEVICE = args.device
    else:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = torch.device('cpu')
    print(DEVICE)

    print('Loading word embeddings:')
    word_indexer = data.WordIndexer(EMB_DIRECTORY)

    # load state dict
    state = torch.load(Path(EXPER_DIRECTORY).joinpath('last.pth'))
    FEATURE_TYPE = state['ft_type']
    
    print('Loading validation dataset:')
    val_annotations, val_videos = utils.load_dataset_info('val', DATASET_DIRECTORY, args.missed_videos)
    val_dataset = data.CustomDataset(val_videos, val_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE, validate=True)
    val_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.ValidateBatchSampler(val_annotations, val_dataset.num_segments_info, iou_threshold=0.5))

    model = models.CALModel(
        pretrained_emb=word_indexer.get_embeddings(),
        visual_input_dim=data.FEATURE_DIM[FEATURE_TYPE]*2+2,
        emb_dim=data.EMBEDDING_DIM, 
    )
    model.load_state_dict(state['model_state_dict'])
    model = model.to(DEVICE)    
    
    # evaluate model with exhaustive search
    metrics = evaluate(model, val_iter, DEVICE)
    print(''.join([f'{name}: {value:.4f}\t' for name, value in metrics.items()]))

