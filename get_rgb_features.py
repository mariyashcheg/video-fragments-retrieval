'''
Retrieve RGB features from pretrained VGG19 for videos
'''

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import os
import argparse
import json
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path


FRAMES_PER_SEC = 5
SEC_PER_SEGMENT = 5
SELECT_FPS = 25

def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

class DiDeMoDataset(Dataset):

    def __init__(self, dataset_info, dataset_dir='.'):
        self.dataset_info = dataset_info
        self.dataset_dir = dataset_dir
        self.mean = torch.Tensor([0.485, 0.456, 0.406]) # ImageNet mean
        self.std = torch.Tensor([0.229, 0.224, 0.225]) # ImageNet std

    def __getitem__(self, index):
        filename = self.dataset_info[index]['video']
        num_segments = self.dataset_info[index]['num_segments']
        vframes, _, vinfo = torchvision.io.read_video(
            filename='{}/{}.mp4'.format(self.dataset_dir, filename),
            pts_unit='sec',
            end_pts=SEC_PER_SEGMENT*num_segments
        )
        if vframes.size(0) > 0: # successed to read videofile:
            
            video_length = vframes.size(0) / vinfo['video_fps']
            # subsample frames (5 frames corresponds to 1 sec of the video)
            if round(video_length) == num_segments*SEC_PER_SEGMENT: # all segments are full (5 sec each)
                step = (vframes.size(0)-1) / (SELECT_FPS*num_segments+1)
            else: # last segment is less than 5 sec
                step = (round((num_segments-1)*SEC_PER_SEGMENT*vinfo['video_fps'])-1) / (SELECT_FPS * (num_segments-1)+1)

            stop = min(vframes.size(0), SELECT_FPS * num_segments * step)-1
            curr_step = 0
            mask = [curr_step]
            while curr_step <= stop - step:
                curr_step += step
                mask.append(int(round(curr_step)))
            mask = torch.Tensor(mask).long()
            vframes = vframes.index_select(dim=0, index=mask)

            # [T, H, W, C] -> [T, C, H, W]
            vframes = vframes.transpose(3, 1).transpose(2, 3)

            # scale to [0, 1]
            vframes = vframes.float().div(255)
            # normalize with ImageNet mean and std
            vframes = vframes.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])

            return dict(
                frames=vframes, 
                video=filename)

        else:
            return dict(
                frames=torch.empty((0,1,1,3)),
                video=filename)


    def __len__(self):
        return len(self.dataset_info)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='videos/')
    parser.add_argument("--features_dir", type=str, default='features/')
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default='vgg19')
    args = parser.parse_args()

    DATASET_DIRECTORY = args.dataset_dir
    FT_DIRECTORY = args.features_dir
    MODEL_TYPE = args.model
    if args.device in ['cpu','cuda']:
        DEVICE = args.device
    else:
        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'

    Path(FT_DIRECTORY).mkdir(exist_ok=True)
    done_videos = [f.stem[len('{}_ft_'.format(MODEL_TYPE)):] for f in
            sorted(Path(FT_DIRECTORY).glob("{}_ft_*.npy".format(MODEL_TYPE)))]
    if Path('missed_videos_features.json').exists():
        with open('missed_videos_features.json', 'r') as f:
            missed_videos = json.load(f)
    else:
        missed_videos = []
    done_videos.extend(missed_videos)
        
    dataset_info = read_json('didemo_video_info.json') # [:len(done_videos)+4]
    dataset_info = [data for data in dataset_info if data['video'] not in done_videos]
    
    # read all videos from DiDeMo dataset
    didemo_dataset = DiDeMoDataset(dataset_info, dataset_dir=DATASET_DIRECTORY)
    dataloader = DataLoader(didemo_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    if MODEL_TYPE == 'vgg19':
        # load pretrained VGG19 net
        model = torchvision.models.vgg19(pretrained=True).to(DEVICE)
        # need features from fc7 (second FC-layer in classifier)
        model.classifier = nn.Sequential(*[model.classifier[i] for i in range(5)])
    elif MODEL_TYPE == 'resnet152':
        # load pretrained ResNet152
        model = torchvision.models.resnet152(pretrained=True).to(DEVICE)
        # need features after avgpool layer (before final FC-head)
        model = nn.Sequential(*(list(model.children())[:-1]))

    # missed_videos = []
    batch_size = 16
    model.eval()
    for data in tqdm(dataloader):
        vframes = data['frames']
        video = data['video'][0]
        if vframes.size(1) > 0: # successed to read videofile    
            # video is a tuple (string, ), so..
            # TODO(maria): dataloader could return dict, so try it
            with torch.no_grad():
                all_frames = vframes.squeeze(0)
                features = []
                for i in range(0, all_frames.shape[0], batch_size):
                    x = all_frames[i: i + batch_size, ...].to(DEVICE)
                    y = model(x).cpu().numpy()
                    features.append(y)
                # join features back
                features = np.concatenate(features, axis=0)
                np.save(Path(FT_DIRECTORY).joinpath('{}_ft_{}'.format(MODEL_TYPE, video)), features)
        else:
            missed_videos.append(video)
    
    with open('missed_videos_features.json', 'w') as f:
        json.dump(missed_videos, f)
