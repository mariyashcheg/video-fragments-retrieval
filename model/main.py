import argparse
import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data
import models
import utils
import evaluate_single

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


class Trainer:

    def __init__(self, train_writer, test_writer, normalize_loss=False, compute_grads=True, device=None, b=0.1, lamb=0.4):
        self.device = device
        self.train_writer = train_writer
        self.test_writer = test_writer
        self.compute_grads = compute_grads
        self.global_step = 0
        self.b = b
        self.lamb = lamb
        self.normalize_loss = normalize_loss

    def train_epoch(self, model, iterator, optimizer, log_prefix=''):
        
        model = model.to(self.device)
        model.train()
        epoch_loss, count_loss = 0, 0
        for batch in tqdm(iterator):
            
            posit_ft = batch['posit'].to(self.device)
            intra_ft = batch['intra'].to(self.device)
            inter_ft = batch['inter'].to(self.device)
            lang_ft = batch['lang'].to(self.device)
            maskp = batch['maskp'].to(self.device)
            maskn = batch['maskn'].to(self.device)

            optimizer.zero_grad()
            posit_emb = model(posit_ft)
            intra_emb = model(intra_ft)
            inter_emb = model(inter_ft)
            lang_emb = model(lang_ft, False, self.device)
            
            loss, n_samples = self.ranking_loss(posit_emb, intra_emb, inter_emb, lang_emb, maskp, maskn)        
            epoch_loss += loss.item()
            count_loss += n_samples
            loss.backward()
            optimizer.step()

            log_entry = dict(
                loss=loss.item() / n_samples
            )
            if self.compute_grads:
                log_entry['grad_norm'] = self.grad_norm(model)

            for name, value in log_entry.items():
                if log_prefix != '':
                    name = log_prefix + '/' + name
                self.train_writer.add_scalar(name, value, global_step=self.global_step)
            self.global_step += 1
        
        return epoch_loss / count_loss

    def test_epoch(self, model, iterator, log_prefix=""):
        
        model = model.to(self.device)
        model.eval()
        epoch_loss = 0
        count_loss = 0        
        for batch in tqdm(iterator):

            posit_ft = batch['posit'].to(self.device)
            intra_ft = batch['intra'].to(self.device)
            inter_ft = batch['inter'].to(self.device)
            lang_ft = batch['lang'].to(self.device)
            maskp = batch['maskp'].to(self.device)
            maskn = batch['maskn'].to(self.device)


            with torch.no_grad():
                posit_emb = model(posit_ft)
                intra_emb = model(intra_ft)
                inter_emb = model(inter_ft)
                lang_emb = model(lang_ft, False, self.device)
                
                loss, n_samples = self.ranking_loss(posit_emb, intra_emb, inter_emb, lang_emb, maskp, maskn)
                epoch_loss += loss.item()
                count_loss += n_samples
        
        log_entry = dict(
            loss=epoch_loss / count_loss
        )
        for name, value in log_entry.items():
            if log_prefix != '':
                name = log_prefix + '/' + name
            self.test_writer.add_scalar(name, value, global_step=self.global_step)
        
        return log_entry['loss']

    def ranking_loss(self, posit_emb, intra_emb, inter_emb, lang_emb, maskp, maskn):

        loss = 0
        n_samples = maskp.max().item()+1

        if self.normalize_loss:
            posit_emb = posit_emb.div(posit_emb.norm(dim=1, keepdim=True) + 1e-5)
            intra_emb = intra_emb.div(intra_emb.norm(dim=1, keepdim=True) + 1e-5)
            inter_emb = inter_emb.div(inter_emb.norm(dim=1, keepdim=True) + 1e-5)
            lang_emb = lang_emb.div(lang_emb.norm(dim=1, keepdim=True) + 1e-5) 

        for i in range(n_samples):
            maskp_i = maskp == i
            maskn_i = maskn == i
            c_posit = F.pairwise_distance(posit_emb[maskp_i], lang_emb[i].repeat(posit_emb[maskp_i].size(0), 1)).mean()
            c_intra = F.pairwise_distance(intra_emb[maskn_i], lang_emb[i].repeat(intra_emb[maskn_i].size(0), 1)).mean()
            c_inter = F.pairwise_distance(inter_emb[maskp_i], lang_emb[i].repeat(inter_emb[maskp_i].size(0), 1)).mean()
            loss += (F.relu(c_posit - c_intra + self.b) + self.lamb*F.relu(c_posit - c_inter + self.b))
        return loss, n_samples

    @staticmethod
    def grad_norm(model):
        grad = 0.0
        count = 0
        for name, tensor in model.named_parameters():
            if tensor.grad is not None:
                grad += tensor.grad.data.norm().cpu().item()
                # grad += torch.sqrt(torch.sum((tensor.grad.data) ** 2))
                count += 1
        return grad / count #.cpu().numpy()


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default=None)
    parser.add_argument("--missed_videos", type=str, default='../')
    parser.add_argument("--normalize_loss", type=utils.str2bool, default=False)
    parser.add_argument("--intra_same_length", type=utils.str2bool, default=True)
    parser.add_argument("--normalize_lang", type=utils.str2bool, default=False)
    parser.add_argument("--pooling", type=str, default='avg')
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
    print(DEVICE, FEATURE_TYPE)

    train_annotations, train_videos = utils.load_dataset_info('train', DATASET_DIRECTORY, MISSED_VIDEOS)
    test_annotations, test_videos = utils.load_dataset_info('test', DATASET_DIRECTORY, MISSED_VIDEOS)

    print('Loading word embeddings:')
    word_indexer = data.WordIndexer(EMB_DIRECTORY)
    print('Loading train dataset:')
    train_dataset = data.CustomDataset(
        train_videos, train_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE, pooling=args.pooling)
        
    print('Loading test dataset:')
    test_dataset = data.CustomDataset(
        test_videos, test_annotations, word_indexer, FT_DIRECTORY, FEATURE_TYPE, pooling=args.pooling)
    
    train_iter = DataLoader(train_dataset, shuffle=False, collate_fn=data.custom_collate, 
        batch_sampler=data.CustomBatchSampler(
            BATCH_SIZE, train_annotations, train_dataset.num_segments_info, same_length=args.intra_same_length))    
    test_iter = DataLoader(test_dataset, shuffle=False, collate_fn=data.custom_collate, 
        batch_sampler=data.CustomBatchSampler(
            BATCH_SIZE, test_annotations, test_dataset.num_segments_info, train=False, same_length=args.intra_same_length))
    
    new_experiment = utils.start_new_experiment(EXPER_DIRECTORY)
    print('\nStart training:')
    trainer = Trainer(
        train_writer=SummaryWriter(new_experiment.joinpath('train_logs')),
        test_writer=SummaryWriter(new_experiment.joinpath('test_logs')),
        normalize_loss=args.normalize_loss,
        compute_grads=True, 
        device=DEVICE, 
        b=0.1, lamb=0.4
    )

    model = models.CALModel(
        pretrained_emb=word_indexer.get_embeddings(),
        visual_input_dim=data.FEATURE_DIM[FEATURE_TYPE]*2+2,
        emb_dim=data.EMBEDDING_DIM, 
        normalize_lang=args.normalize_lang
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.95)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_loss = None
    for epoch in range(N_EPOCHES):

        train_loss = trainer.train_epoch(model, train_iter, optimizer)
        test_loss = trainer.test_epoch(model, test_iter)
        scheduler.step()

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            loss=test_loss,
            params={param: value for param, value in vars(args).items() 
                            if param in ['device','feature_type','normalize_loss','normalize_lang','pooling']},
            global_step=trainer.global_step,
        )
        torch.save(state, new_experiment.joinpath('last.pth'))
        if best_loss is None:
            best_loss = test_loss
        elif test_loss < best_loss:
            best_loss = test_loss
            torch.save(state, new_experiment.joinpath('best.pth'))
        print(f'Epoch: {epoch+1:02},\tTrain Loss: {train_loss:.4f},\tTest Loss: {test_loss:.4f}')