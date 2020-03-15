import argparse
import json
import numpy as np
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data
import models

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

def start_new_experiment(exper_dir):
    if Path(exper_dir).exists() and Path(exper_dir).is_dir():
        experiments = []
        for d in Path(exper_dir).iterdir():
            n = re.fullmatch(r'\d+', d.name)
            if d.is_dir() and n is not None:
    parser.add_argument("--feature_type", type=str, default=None)
                experiments.append(int(n.group()))
        experiments = sorted(experiments)        
        new_experiment = str(experiments[-1]+1)
    else:
        Path(exper_dir).mkdir(exist_ok=False)
        new_experiment = '0'
    new_exper_dir = Path(exper_dir).joinpath(new_experiment)
    Path(new_exper_dir).mkdir(exist_ok=False)
    return new_exper_dir


class Trainer:

    def __init__(self, train_writer=None, eval_writer=None, compute_grads=True, device=None, b=0.1, lamb=0.4):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        self.compute_grads = compute_grads
        self.global_step = 0
        self.b = b
        self.lamb = lamb

    def train_epoch(self, model, iterator, optimizer, log_prefix=''):
        
        model = model.to(self.device)
        model.train()
        for batch in tqdm(iterator):
            
            posit_ft = batch['posit'][0].to(self.device)
            intra_ft = batch['intra'][0].to(self.device)
            inter_ft = batch['inter'][0].to(self.device)
            lang_ft = batch['lang'][0].to(self.device)

            optimizer.zero_grad()
            posit_emb, intra_emb, inter_emb, lang_emb = model(
                posit_ft, intra_ft, inter_ft, lang_ft, self.device)
            
            loss = self.ranking_loss(posit_emb, intra_emb, inter_emb, lang_emb)        
            loss.backward()
            optimizer.step()

            log_entry = dict(
                loss=loss.item()
            )
            if self.compute_grads:
                log_entry['grad_norm'] = self.grad_norm(model)

            for name, value in log_entry.items():
                if log_prefix != '':
                    name = log_prefix + '/' + name
                self.train_writer.add_scalar(name, value, global_step=self.global_step)
            self.global_step += 1

    def eval_epoch(self, model, iterator, log_prefix=""):
        
        model = model.to(self.device)
        model.eval()
        epoch_loss = 0        
        for batch in tqdm(iterator):

            posit_ft = batch['posit'][0].to(self.device)
            intra_ft = batch['intra'][0].to(self.device)
            inter_ft = batch['inter'][0].to(self.device)
            lang_ft = batch['lang'][0].to(self.device)

            with torch.no_grad():
                posit_emb, intra_emb, inter_emb, lang_emb = model(
                    posit_ft, intra_ft, inter_ft, lang_ft, self.device)
                
                loss = self.ranking_loss(posit_emb, intra_emb, inter_emb, lang_emb)
                epoch_loss += loss.item()
        
        log_entry = dict(
            loss=epoch_loss / len(iterator)
        )
        for name, value in log_entry.items():
            if log_prefix != '':
                name = log_prefix + '/' + name
            self.eval_writer.add_scalar(name, value, global_step=self.global_step)
        
        return log_entry['loss']

    def ranking_loss(self, posit_emb, intra_emb, inter_emb, lang_emb):

        c_posit = F.pairwise_distance(posit_emb, lang_emb.repeat(posit_emb.size(0), 1)).mean()
        c_intra = F.pairwise_distance(intra_emb, lang_emb.repeat(intra_emb.size(0), 1)).mean()
        c_inter = F.pairwise_distance(inter_emb, lang_emb.repeat(inter_emb.size(0), 1)).mean()

        return F.relu(c_posit - c_inter + self.b) + self.lamb*F.relu(c_posit - c_intra + self.b)

    @staticmethod
    def grad_norm(model):
        grad = 0.0
        count = 0
        for name, tensor in model.named_parameters():
            if tensor.grad is not None:
                grad += torch.sqrt(torch.sum((tensor.grad.data) ** 2))
                count += 1
        return grad.cpu().numpy() / count



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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--feature_type", type=str, default=None)
    args = parser.parse_args()

    FT_DIRECTORY = args.features_dir
    EMB_DIRECTORY = args.embedding_dir
    DATASET_DIRECTORY = args.dataset_dir
    EXPER_DIRECTORY = args.experiment_dir
    N_EPOCHES = args.n_epoches

    if args.feature_type is in data.FEATURE_DIM.keys():
        FEATURE_TYPE = args.feature_type
    else:
        raise ValueError("Unknown feature type '{}'".format(args.feature_type))
        
    if args.device in ['cpu','cuda']:
        DEVICE = args.device
    else:
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
    
    # no features for these videos
    if Path('../missed_videos_features.json').exists():
        with open('../missed_videos_features.json', 'r') as f:
            missed_videos = json.load(f)
            print('Loaded missed video list')
    else:
        missed_videos = []

    train_data = [video for video in read_json(Path(DATASET_DIRECTORY).joinpath('train_data.json'))
        if video['video'] not in missed_videos]
    test_data = [video for video in read_json(Path(DATASET_DIRECTORY).joinpath('test_data.json'))
        if video['video'] not in missed_videos]
    val_data = [video for video in read_json(Path(DATASET_DIRECTORY).joinpath('val_data.json'))
        if video['video'] not in missed_videos]

    word_indexer = data.WordIndexer(EMB_DIRECTORY)
    train_dataset = data.CustomDataset(train_data, word_indexer, FT_DIRECTORY, FEATURE_TYPE)
    test_dataset = data.CustomDataset(test_data, word_indexer, FT_DIRECTORY, FEATURE_TYPE)
    # val_dataset = data.CustomDataset(val_data, word_indexer, FT_DIRECTORY, FEATURE_TYPE)

    train_iter = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=True)
    # val_iter = DataLoader(val_dataset, batch_size=1, shuffle=True)

    new_experiment = start_new_experiment(EXPER_DIRECTORY)
    trainer = Trainer(
        train_writer=SummaryWriter(new_experiment.joinpath('train_logs')),
        eval_writer=SummaryWriter(new_experiment.joinpath('test_logs')),
        compute_grads=True, 
        device=DEVICE, 
        b=0.1, lamb=0.4
    )

    model = models.CALModel(
        pretrained_emb=word_indexer.get_embeddings(),
        visual_input_dim=data.FEATURE_DIM[FEATURE_TYPE]*2+2,
        emb_dim=data.EMBEDDING_DIM, 
    )

    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.95)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(N_EPOCHES):

        trainer.train_epoch(model, train_iter, optimizer)
        test_loss = trainer.eval_epoch(model, test_iter)
        scheduler.step()

        state = dict(
            epoch=epoch,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            loss=test_loss,
            global_step=trainer.global_step,
        )

        print(f'Epoch: {epoch+1:02},\tVal. Loss: {test_loss:.3f}')
        torch.save(state, new_experiment.joinpath('last.pth'))
