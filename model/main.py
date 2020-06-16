import argparse
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import data
import models
import utils

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from transformers import BertTokenizer, BertModel

plt.switch_backend('agg')


class Trainer:

    def __init__(self, train_writer=None, test_writer=None, val_writer=None, normalize_loss=False, 
        compute_grads=True, device=None, b=0.1, lamb=0.4):

        self.device = device
        self.train_writer = train_writer
        self.test_writer = test_writer
        self.val_writer = val_writer
        self.compute_grads = compute_grads
        self.global_step = 0
        self.b = b
        self.lamb = lamb
        self.normalize_loss = normalize_loss

    def train_epoch(self, model, iterator, optimizer):
        
        model = model.to(self.device)
        model.train()
        epoch_loss, count_loss = 0, 0
        for batch in iterator:
            
            posit_ft = batch['posit']
            negat_ft = batch['negat']
            endp_posit = batch['endp_posit'].to(self.device)
            endp_negat = batch['endp_negat'].to(self.device)
            lang_ft = batch['lang'].to(self.device)
            len_seq = batch['len_seq'].to(self.device)

            optimizer.zero_grad()

            posit_emb = model(posit_ft, True, endpoints=endp_posit, device=self.device)
            intra_emb = model(posit_ft, True, endpoints=endp_negat, device=self.device)
            inter_emb = model(negat_ft, True, endpoints=endp_posit, device=self.device)
            lang_emb = model(lang_ft, False, len_seq=len_seq)
                
            loss, n_samples = self.ranking_loss(posit_emb, intra_emb, inter_emb, lang_emb)
            epoch_loss += loss.item()
            count_loss += n_samples
            loss.backward()
            optimizer.step()

            log_entry = dict(
                loss=loss.item() / n_samples
            )
            if self.compute_grads:
                log_entry['grad_norm'] = utils.grad_norm(model)

            for name, value in log_entry.items():
                self.train_writer.add_scalar(name, value, global_step=self.global_step)
            self.global_step += 1
        
        return epoch_loss / count_loss

    def test_epoch(self, model, iterator, lr=0):
        
        model = model.to(self.device)
        model.eval()
        epoch_loss = 0
        count_loss = 0        
        for batch in iterator:

            posit_ft = batch['posit']
            negat_ft = batch['negat']
            endp_posit = batch['endp_posit'].to(self.device)
            endp_negat = batch['endp_negat'].to(self.device)
            lang_ft = batch['lang'].to(self.device)
            len_seq = batch['len_seq'].to(self.device)

            with torch.no_grad():
                posit_emb = model(posit_ft, True, endpoints=endp_posit, device=self.device)
                intra_emb = model(posit_ft, True, endpoints=endp_negat, device=self.device)
                inter_emb = model(negat_ft, True, endpoints=endp_posit, device=self.device)
                lang_emb = model(lang_ft, False, len_seq=len_seq)
                
                loss, n_samples = self.ranking_loss(posit_emb, intra_emb, inter_emb, lang_emb)
                epoch_loss += loss.item()
                count_loss += n_samples
        
        log_entry = dict(
            loss=epoch_loss / count_loss,
            lr=lr
        )
        for name, value in log_entry.items():
            self.test_writer.add_scalar(name, value, global_step=self.global_step)
        
        scalars = dict(
            video=posit_emb.norm(dim=1).mean().cpu().item(),
            language=lang_emb.norm(dim=1).mean().cpu().item()
        )
        self.test_writer.add_scalars('embedding_norm', scalars, global_step=self.global_step)
        
        return log_entry['loss']
    
    def validate_epoch(self, model, video_iterator, lang_iterator, annotations, 
        size=250, iou_thresholds=[0.5, 0.7], atk=[1,10,100], new_experiment=None):
        
        videos = {}
        videos_index = {}
        model.eval()
        video_count = 1
        # compute embeddings for all clips in each unique video in dataset
        for vi, batch in enumerate(video_iterator):
            with torch.no_grad():
                videos[batch['video']] = model(
                    [batch['feature']], True, 
                    endpoints=batch['endpoints'].unsqueeze(0).to(self.device), 
                    device=self.device).squeeze(0)
                if size == -1:
                    videos_index[batch['video']] = vi
        if size == -1:
            video_count += vi
            with open(Path(new_experiment).joinpath('videos_index_clip.json'), 'w') as f:
                json.dump(videos_index, f)
        
        custom = defaultdict(lambda: defaultdict(list))
        recipr_rank = defaultdict(list)
        median_rank = defaultdict(list)
        median_rank_all = defaultdict(list)
        # true_posit = defaultdict(lambda: defaultdict(lambda: 0))
        # total_relevant = defaultdict(lambda: 0)

        moments = lang_iterator.batch_sampler.moments
        annot_index = {}
        heatmap = []

        # for num_seg, mnts in lang_iterator.batch_sampler.moments.items():
        #     moments[num_seg] = [tuple((start_t, end_t)) for start_t, end_t in mnts if end_t > start_t]
        thr_range = [i/10 for i in range(5,10)] if size == -1 else iou_thresholds
        # compute embeddings for all queries in dataset
        for li, batch in enumerate(lang_iterator):
            with torch.no_grad():
                lang_emb = model(
                    batch['feature'].to(self.device), False, len_seq=batch['len_seq'].to(self.device))
            if size == -1:    
                annot_index[batch['annot_id']] = li

            # ground truth: 1 if IoU between moment and at least 2 manual annotations is greater the threshold
            # for incorrect videos all IoU scores are 0
            heatmap_query = np.zeros((video_count, 6))
            distances, predicts = [], defaultdict(list)
            for video in videos.keys():
                video_emb = videos[video]
                num_segments = video_emb.size(0)
                # for each query compute distance between query and clip embeddings
                dist = F.pairwise_distance(video_emb, lang_emb.repeat(num_segments, 1))
                if size == -1:
                    dist_np = dist.detach().cpu().numpy().flatten()
                    heatmap_query[videos_index[video], :dist_np.shape[0]] = dist_np
                # for each possible moment in video compute distance between moment and query
                # as sum of distances between clips in the moment and query (precomputed on prev step)
                # (15 possible moments for video of 5 clips, 21 - for 6 clips)
                for start_t, end_t in moments[num_segments]:
                    distances.append(dist.index_select(0, torch.arange(start_t, end_t+1).to(self.device)).mean().item())
                    if video == batch['video']:
                        ious = utils.get_iou(annotations[batch['annot_id']]['times'], start_t, end_t)
                        for iou_thr in thr_range:
                            predicts[iou_thr].append(int(( ious >= iou_thr ).sum() >= 2))
                    else:
                        for iou_thr in thr_range:
                            predicts[iou_thr].append(0)

            if size == -1:
                heatmap.append(heatmap_query)
            # sort all moments from all videos by distances
            for iou_thr in thr_range:
                predicts[iou_thr] = np.array(predicts[iou_thr], dtype=int)[np.argsort(distances)]
                # index position of first correct retrieval
                rank = np.where(predicts[iou_thr] == 1)[0]+1
                # if len(rank) > 0:
                # rank = rank[0] + 1
                # else:
                #     rank = predicts[iou_thr].shape[0]
                #     print(annotations[batch['annot_id']])
                # total_relevant[iou_thr] += predicts[iou_thr].sum()

                if iou_thr in iou_thresholds:
                    median_rank[iou_thr].append(rank[0])
                    median_rank_all[iou_thr].extend(rank.tolist())
                    recipr_rank[iou_thr].append(1 / rank[0])

                for k in atk:
                    topk = predicts[iou_thr][:k]

                    # if size == -1:
                    #     true_posit[iou_thr][k] += topk.sum()

                    # if iou_thr in iou_thresholds:
                    # Custom Recall@K: 1 if correct moment is among in top-K moments, 0 otherwise
                    custom[iou_thr][k].append(int(rank[0] <= k))
                            
            if (li == size - 1):
                break
        
        if size == -1:
            with open(Path(new_experiment).joinpath('annot_index_clip.json'), 'w') as f:
                json.dump(annot_index, f)

            np.save(Path(new_experiment).joinpath('heatmap_clip'), np.stack(heatmap))

        scalars = {}
        for iou_thr, values in custom.items():
            scalars.update({f'{k}_IoU0{round(iou_thr*10)}': np.mean(v) for k,v in values.items()})
        self.val_writer.add_scalars('CustomRecall', scalars, global_step=self.global_step)

        scalars = {f'IoU0{round(iou_thr*10)}': np.median(values) for iou_thr, values in median_rank.items()}
        self.val_writer.add_scalars('MedianRank', scalars, global_step=self.global_step)

        scalars = {f'IoU0{round(iou_thr*10)}': np.median(values) for iou_thr, values in median_rank_all.items()}
        self.val_writer.add_scalars('MedianRankAll', scalars, global_step=self.global_step)

        scalars = {f'IoU0{round(iou_thr*10)}': np.mean(values) for iou_thr, values in recipr_rank.items()}
        self.val_writer.add_scalars('MeanReciprocalRank', scalars, global_step=self.global_step)

        histogram = {}
        if size == -1:
            for iou_thr, values in median_rank.items():
                fig = plt.figure(figsize=(12,6))
                plt.hist(values, bins=np.arange(0,np.max(values),100))
                plt.grid()
                histogram[f'MedianRank1_IoU0{round(iou_thr*10)}'] = values
                self.val_writer.add_figure(f'MedianRank1_IoU0{round(iou_thr*10)}', fig, global_step=self.global_step)
        # pr_curve = defaultdict(lambda: defaultdict(list))
        # if size == -1:
        #     for k in atk:
        #         for iou_thr in thr_range:
        #             pr_curve['precision'][k].append( true_posit[iou_thr][k] / (k*(li+1)) )
        #             pr_curve['recall'][k].append( true_posit[iou_thr][k] / total_relevant[iou_thr] )
        #         fig = plt.figure(figsize=(12,6))
        #         plt.plot(pr_curve['recall'][k], pr_curve['precision'][k])
        #         plt.grid()
        #         self.val_writer.add_figure(f'pr_curve@{k}', fig, global_step=self.global_step)
        
        return histogram

    def ranking_loss(self, posit_emb, intra_emb, inter_emb, lang_emb):

        bs, lm, dim = posit_emb.size()
        posit_emb = posit_emb.view(-1, dim)
        intra_emb = intra_emb.view(-1, dim)
        inter_emb = inter_emb.view(-1, dim)
        lang_emb = lang_emb.unsqueeze(1).repeat(1, lm, 1).view(-1, dim)

        if self.normalize_loss:
            posit_emb = posit_emb.div(posit_emb.norm(dim=1, keepdim=True) + 1e-5)
            intra_emb = intra_emb.div(intra_emb.norm(dim=1, keepdim=True) + 1e-5)
            inter_emb = inter_emb.div(inter_emb.norm(dim=1, keepdim=True) + 1e-5)
            lang_emb = lang_emb.div(lang_emb.norm(dim=1, keepdim=True) + 1e-5) 

        c_posit = F.pairwise_distance(posit_emb, lang_emb).view(bs, lm).mean(axis=-1)
        c_intra = F.pairwise_distance(intra_emb, lang_emb).view(bs, lm).mean(axis=-1)
        c_inter = F.pairwise_distance(inter_emb, lang_emb).view(bs, lm).mean(axis=-1)
        loss = F.relu(c_posit - c_intra + self.b) + self.lamb*F.relu(c_posit - c_inter + self.b)

        return loss.sum(), bs


def cyclic_learning_rate(stepsize, min_lr=5e-5, max_lr=1e-3, scaler_mode='clr', gamma=0.95):
    if scaler_mode == 'clr':
        scaler = lambda x: 1
    elif scaler_mode == 'exp_clr':
        scaler = lambda x: gamma ** x

    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    def relative(it, stepsize):
        cycle = np.floor(1 + it / float(2 * stepsize))
        x = abs(it / float(stepsize) - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


if __name__ == '__main__':

    torch.random.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default='features/')
    parser.add_argument("--embedding_dir", type=str, default='glove_pretrained/')
    parser.add_argument("--dataset_dir", type=str, default='dataset/')
    parser.add_argument("--experiment_dir", type=str, default='experiment/')
    parser.add_argument("--n_epoches", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default=None)
    parser.add_argument("--missed_videos", type=str, default='../')
    parser.add_argument("--normalize_loss", type=utils.str2bool, default=False)
    parser.add_argument("--intra_same_length", type=utils.str2bool, default=True)
    parser.add_argument("--aggregation", type=str, default='conv_pooling')
    parser.add_argument("--pooling", type=str, default='avg')
    parser.add_argument("--val_size", type=int, default=250)
    args = parser.parse_args()

    FT_DIRECTORY = args.features_dir
    EMB_DIRECTORY = args.embedding_dir
    DATASET_DIRECTORY = args.dataset_dir
    EXPER_DIRECTORY = args.experiment_dir
    N_EPOCHES = args.n_epoches
    BATCH_SIZE = args.batch_size
    MISSED_VIDEOS = args.missed_videos

    if args.feature_type in data.FEATURE_DIM.keys():
        FEATURE_TYPE = args.feature_type
    else:
        raise ValueError(f"Unknown feature type '{args.feature_type}'")
        
    if args.device in ['cpu','cuda']:
        DEVICE = args.device
    else:
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = torch.device('cpu')
    print(DEVICE, FEATURE_TYPE, args.aggregation, args.pooling)

    train_annotations, train_videos = utils.load_dataset_info('train', DATASET_DIRECTORY, MISSED_VIDEOS)
    test_annotations, test_videos = utils.load_dataset_info('test', DATASET_DIRECTORY, MISSED_VIDEOS)

    print('Loading word embeddings:')
    word_indexer = data.WordIndexer(EMB_DIRECTORY)
    
    print('Loading train dataset:')
    train_dataset = data.CustomDataset(
        train_videos, train_annotations, FT_DIRECTORY, FEATURE_TYPE, word_indexer=word_indexer)

    print('Loading test dataset:')
    test_dataset = data.CustomDataset(
        test_videos, test_annotations, FT_DIRECTORY, FEATURE_TYPE, word_indexer=word_indexer)
        
    train_iter = DataLoader(train_dataset, shuffle=False, collate_fn=data.custom_collate, 
        batch_sampler=data.CustomBatchSampler(
            BATCH_SIZE, train_dataset.annotations, train_dataset.num_segments_info, train_dataset.annot_len_info))    
    test_iter = DataLoader(test_dataset, shuffle=False, collate_fn=data.custom_collate, 
        batch_sampler=data.CustomBatchSampler(
            BATCH_SIZE, test_dataset.annotations, test_dataset.num_segments_info, test_dataset.annot_len_info))
    
    print('Loading val dataset:')
    val_dataset = data.CustomDataset(test_videos, test_annotations, FT_DIRECTORY, FEATURE_TYPE, 
        word_indexer=word_indexer, validate=True)
    val_video_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.VideoBatchSampler(test_videos, val_dataset.num_segments_info))
    val_lang_iter = DataLoader(val_dataset, shuffle=False, collate_fn=data.validate_collate, 
        batch_sampler=data.LanguageBatchSampler(val_dataset.annotations, val_dataset.num_segments_info))
    
    new_experiment = utils.start_new_experiment(EXPER_DIRECTORY)
    print('\nStart training:')
    trainer = Trainer(
        train_writer=SummaryWriter(new_experiment.joinpath('train_logs')),
        test_writer=SummaryWriter(new_experiment.joinpath('test_logs')),
        val_writer=SummaryWriter(new_experiment.joinpath('val_logs')),
        normalize_loss=args.normalize_loss,
        compute_grads=True, 
        device=DEVICE,
        b=0.1, lamb=0.4
    )

    model = models.CALModel(
        visual_input_dim=data.FEATURE_DIM[FEATURE_TYPE],
        embedding_dim=data.EMBEDDING_DIM, 
        aggr_type=args.aggregation, 
        pooling=args.pooling,
        max_query_len=train_dataset.max_query_len
    )

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-3)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.95)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = cyclic_learning_rate(5)
    
    best_loss = None
    for epoch in tqdm(range(N_EPOCHES)):

        lr = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = trainer.train_epoch(model, train_iter, optimizer)
        # for param_group in optimizer.param_groups:
        #     lr = param_group['lr']
        test_loss = trainer.test_epoch(model, test_iter, lr)
        if epoch != N_EPOCHES - 1:
            _ = trainer.validate_epoch(model, val_video_iter, val_lang_iter, val_dataset.annotations, size=args.val_size)
        # scheduler.step()

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            loss=test_loss,
            params={param: value for param, value in vars(args).items() 
                            if param in ['device','feature_type','normalize_loss','aggregation','pooling']},
            global_step=trainer.global_step,
        )
        torch.save(state, new_experiment.joinpath('last.pth'))
        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss:
            best_loss = test_loss
            torch.save(state, new_experiment.joinpath('best.pth'))
        # print(f'\nEpoch: {epoch+1:02},\tTrain Loss: {train_loss:.4f},\tTest Loss: {test_loss:.4f}')
    
    metrics = trainer.validate_epoch(model, val_video_iter, val_lang_iter, test_annotations, size=-1, new_experiment=new_experiment)
    with open(Path(new_experiment).joinpath('hist_median_rank.json'), 'w') as f:
        json.dump(metrics, f)